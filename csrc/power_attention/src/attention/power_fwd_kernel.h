/******************************************************************************
 * Copyright (c) 2024, Sean Zhang.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"
#include "power_utils.h"
#include "softmax.h"
#include "mask.h"
#include "rotary.h"
#include "gating.h"

namespace power {

using namespace cute;

#define PRINT_CONDITION (tidx == 0 && m_block == 0 && bidh == 0 && bidb == 0)

template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_softcap(Tensor<Engine, Layout> &tensor, const float softcap){
    #pragma unroll
    for (int i = 0; i < size(tensor); ++i) {
        tensor(i) = cutlass::fast_tanh(tensor(i) * softcap);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template<typename ElementAccum, typename Params, int kBlockM, bool Is_even_MN>
__forceinline__ __device__ auto get_lse_tile(const Params &params, const int bidb, const int bidh, const int m_block, const BlockInfo</*Varlen=*/!Is_even_MN> &binfo) {
        // When params.unpadded_lse is false, LSE is written as (b, h, seqlen_q) - this is non-variable seqlen path.
        // Otherwise, when params.seqlenq_ngroups_swapped is true, it is written as (h, seqlen_q, b) to account for seqlen_q <-> h swapping trick.
        // Otherwise, it's written as (h, b, seqlen_q).
        const bool varlen_q = params.unpadded_lse && !params.seqlenq_ngroups_swapped;
        auto lse_offset = varlen_q ? binfo.q_offset(params.seqlen_q, 1, bidb) : 0;
        auto gmem_ptr_lse = make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr) + lse_offset);

        auto lse_shape = varlen_q ? make_shape(1, params.h, params.total_q) : make_shape(params.b, params.h, params.seqlen_q);
        auto lse_stride = params.seqlenq_ngroups_swapped ? make_stride(1, params.seqlen_q * params.b, params.b) : (
            params.unpadded_lse ? make_stride(params.h * params.total_q, params.total_q, 1) :  make_stride(params.h * params.seqlen_q, params.seqlen_q, 1)
            );

        auto lse_layout = make_layout(lse_shape, lse_stride);
        Tensor mLSE = make_tensor(gmem_ptr_lse, lse_layout);
        auto mLSE_slice = varlen_q ? mLSE(0, bidh, _) : mLSE(bidb, bidh, _);
        return local_tile(mLSE_slice, Shape<Int<kBlockM>>{}, make_coord(m_block));
}


template<typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Return_softmax, bool IsGating, bool FlashEquivalent, bool NormalSpace, int Deg, typename Params>
inline __device__ void compute_attn_1rowblock(const Params &params, const int bidb, const int bidh, const int m_block) {

    // Math:
    // Forward:
    // M = Causal Mask
    // S = QK^T * M
    // T = p * log(abs(S) + ε) - log_stabilizer
    // Z = T + (G_Q @ 1^T - 1 @ G_K^T)
    // P = exp(Z - Z_rowmax)
    // Y = P @ V
    // y = P @ 1

    // Options:
    // Is_Q_in_regs: put Q in registers instead of smem, reducing smem pressure at the cost of register pressure
    // Is_G_Q_in_regs: not implemented yet

    // Main Steps:
    // Prologue:
    //       * Start loading Q
    //       * Start loading GQ
    //       * Start loading K[-1]
    //       * Start loadin

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr bool is_bf16 = std::is_same_v<Element, cutlass::bfloat16_t>;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;
    // m_block is 0-indexed
    const bool is_last_m_block = m_block == (cute::ceil_div(binfo.actual_seqlen_q, kBlockM) - 1);

    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // Otherwise we might read OOB elements from gK and gV.
    if ((Is_causal || !Is_even_MN) && n_block_max == 0) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        power::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        return;
    }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    const index_t offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;
    const index_t offset_norm = bidb * params.norm_batch_stride + m_block * kBlockM * params.norm_row_stride + bidh * params.norm_head_stride;
    const index_t offset_rowmax = bidb * params.rowmax_batch_stride + m_block * kBlockM * params.rowmax_row_stride + bidh * params.rowmax_head_stride;
    const index_t offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb) + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb) + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb) + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t offset_gq = binfo.q_offset(params.g_batch_stride, params.g_row_stride, bidb) + m_block * kBlockM * params.g_row_stride + bidh * params.g_head_stride;
    const index_t offset_gk = binfo.k_offset(params.g_batch_stride, params.g_row_stride, bidb) + (n_block_max - 1) * kBlockN * params.g_row_stride + bidh * params.g_head_stride;
    const index_t offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
                                          + offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
                                          + offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));

    // A workaround to turn off gating without conditional types
    Tensor gGQ = make_tensor(make_gmem_ptr(reinterpret_cast<float *>(IsGating ? params.log_g_q_ptr : params.q_ptr) + offset_gq),
                            Shape<Int<kBlockM>>{},
                            make_stride(params.g_row_stride));
    Tensor gGK = make_tensor(make_gmem_ptr(reinterpret_cast<float *>(IsGating ? params.log_g_k_ptr : params.k_ptr) + offset_gk),
                            Shape<Int<kBlockN>>{},
                            make_stride(params.g_row_stride));
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));
    Tensor gNorm = make_tensor(make_gmem_ptr(reinterpret_cast<float *>(params.norm_ptr) + offset_norm),
                               Shape<Int<kBlockM>>{},
                               make_stride(params.norm_row_stride));
    Tensor gRowmax = make_tensor(make_gmem_ptr(reinterpret_cast<float *>(params.rowmax_ptr) + offset_rowmax),
                               Shape<Int<kBlockM>>{},
                               make_stride(params.rowmax_row_stride));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    // Careful we're using the same smem for sQ and sK | sV if Is_Q_in_regs;
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Is_Q_in_regs ? 0 : size(sQ)),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
    Tensor sGQ = make_tensor(make_smem_ptr(reinterpret_cast<float *>(&(*sV.data()) + size(sV))), typename Kernel_traits::SmemLayoutGQ{});
    Tensor sGK = make_tensor(make_smem_ptr(reinterpret_cast<float *>(&(*sGQ.data()) + size(sGQ))), typename Kernel_traits::SmemLayoutGK{});


    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopyLogG gmem_tiled_copy_LogG;
    auto gmem_thr_copy_LogG = gmem_tiled_copy_LogG.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
    Tensor tGQgGQ = gmem_thr_copy_LogG.partition_S(gGQ);  // (KCPY, KCPY_M)
    Tensor tGQsGQ = gmem_thr_copy_LogG.partition_D(sGQ);
    Tensor tGKgGK = gmem_thr_copy_LogG.partition_S(gGK);  // (KCPY, KCPY_N, nblocksN)
    Tensor tGKsGK = gmem_thr_copy_LogG.partition_D(sGK); 

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor tSgS  = thr_mma.partition_C(gP);

#if DEBUG_POWER_FWD
    Tensor sP = make_tensor(make_smem_ptr(reinterpret_cast<ElementAccum *>(&(*sGK.data()) + (IsGating ? size(sGK) : 0))), typename Kernel_traits::SmemLayoutP{});
    Tensor tPsP = thr_mma.partition_C(sP);
#endif


    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o);

    //
    // Copy Atom retiling
    //
    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);


    //
    // PREDICATES
    //
    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    auto load_GQ = [&](const bool Clear_OOB_MN=true) {
        if constexpr (!IsGating) {
            return;
        }
        Tensor cGQ = make_identity_tensor(make_shape(size<0>(sGQ)));    // (BLK_M) -> (blk_m)
        Tensor tGQcGQ = gmem_thr_copy_LogG.partition_D(cGQ);
        if (Clear_OOB_MN) {
            power::copy1d<false, true, kBlockM>(gmem_tiled_copy_LogG, tGQgGQ, tGQsGQ, tGQcGQ, std::min(binfo.actual_seqlen_q - m_block * kBlockM, kBlockM));
        } else {
            power::copy1d<false, false, kBlockM>(gmem_tiled_copy_LogG, tGQgGQ, tGQsGQ, tGQcGQ, std::min(binfo.actual_seqlen_q - m_block * kBlockM, kBlockM));
        }
    };
    auto load_GK = [&](const int n_block, const bool Clear_OOB_MN=true) {
        if constexpr (!IsGating) {
            return;
        }
        Tensor cGK = make_identity_tensor(make_shape(size<0>(sGK)));    // (BLK_N) -> (blk_n)
        Tensor tGKcGK = gmem_thr_copy_LogG.partition_D(cGK);
        if (Clear_OOB_MN) {
            power::copy1d<false, true, kBlockN>(gmem_tiled_copy_LogG, tGKgGK, tGKsGK, tGKcGK, std::min(binfo.actual_seqlen_k - n_block * kBlockN, kBlockN));
        } else {
            power::copy1d<false, false, kBlockN>(gmem_tiled_copy_LogG, tGKgGK, tGKsGK, tGKcGK, std::min(binfo.actual_seqlen_k - n_block * kBlockN, kBlockN));
        }
        tGKgGK.data() = tGKgGK.data() + index_t(-kBlockN * params.g_row_stride);
    };
    auto load_K = [&](const int n_block, const bool NO_OOB=true, const bool Clear_OOB_MN=true) {
        if (Is_even_K && NO_OOB) {
            cute::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
        } else {
            if (Clear_OOB_MN) {
                power::copy<Is_even_MN, Is_even_K, true>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV,
                                       binfo.actual_seqlen_k - n_block * kBlockN);
            } else {
                power::copy<Is_even_MN, Is_even_K, false>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN);
            }
        }
        tKgK.data() = tKgK.data() + index_t(-kBlockN * params.k_row_stride);
    };
    auto load_V = [&](const int n_block, const bool NO_OOB=true, const bool Clear_OOB_MN=true) {
        if (Is_even_K && NO_OOB) {
            cute::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
        } else {
            if (Clear_OOB_MN) {
                power::copy<Is_even_MN, Is_even_K, true>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN);
            } else {
                power::copy<Is_even_MN, Is_even_K, false>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN);
            }
        }
        tVgV.data() = tVgV.data() + index_t(-kBlockN * params.v_row_stride);
    };
    auto load_Q = [&](const bool NO_OOB=true, const bool Clear_OOB_MN=true) {
        if (Is_even_K && NO_OOB) {
            cute::copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);
        } else {
            if (Clear_OOB_MN) {
                power::copy<Is_even_MN, Is_even_K, true>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);
            } else {
                power::copy<Is_even_MN, Is_even_K, false>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);

            }
        }
    };


    // debugger
    #define DEBUG_PRINT(acc, name) \
        if (true) { \
            cute::copy(acc, tPsP); \
            __syncthreads(); \
            if PRINT_CONDITION { \
                printf("power fwd KERNEL: m_block = %d, n_block = %d, %s:\n", m_block, n_block, name); \
                print_tensor(sP); \
                printf("\n"); \
            } \
            __syncthreads(); \
        }

#ifdef DEBUG_POWER_FWD
    if PRINT_CONDITION {
        printf("power fwd KERNEL: m_block = %d, n_block_max = %d, is_last_m_block = %d\n", m_block, n_block_max, is_last_m_block);
    }
#endif

    // Prologue
    int n_block = n_block_max - 1;

    // Load Q
    // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    load_Q(/*NO_OOB=*/!is_last_m_block, /*Clear_OOB_MN=*/false);
    if (Kernel_traits::Is_Q_in_regs) { cute::cp_async_fence(); }

    // load GQ
    // No need to clear the sGQ because we'll only write out valid outputs
    load_GQ(/*Clear_OOB_MN=*/false);
    cute::cp_async_fence();

    // put Q in register optionally
    if (Kernel_traits::Is_Q_in_regs) {
        power::cp_async_wait<1>();
        __syncthreads();
#ifdef DEBUG_POWER_FWD
        if PRINT_CONDITION {
            printf("preload sQ: \n");
            print_tensor(sQ);
            print("\n");
        }
        __syncthreads();
#endif
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    // load K
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    load_K(n_block, /*NO_OOB=*/!is_last_m_block, /*Clear_OOB_MN=*/false);
    cute::cp_async_fence();

    // load GK
    // No need to clear the sGK because they will be masked out
    load_GK(n_block, /*Clear_OOB_MN=*/false);
    cute::cp_async_fence();

    power::Mask<Is_causal, /*Is_local=*/false, /*Has_alibi=*/false> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right);

    // Register for storing y/norm/rowsum of P
    using acc_tensor = decltype(partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}));
    auto acc_layout = acc_tensor{}.layout();
    using rowcol_layout = decltype(power::convert_layout_acc_rowcol(acc_layout));
    Tensor acc_norm = make_tensor<ElementAccum>(Shape<Int<size<0>(rowcol_layout{})>>{});
    clear(acc_norm);

    power::SymPower<2 * size<1>(acc_layout), is_bf16, Deg> sym_power;

    // For performance reason, we separate out two kinds of iterations:
    // those that need masking on S, and those that don't.
    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.

    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
    constexpr int n_masking_steps = (!Is_causal)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);

#ifdef DEBUG_POWER_FWD
    if PRINT_CONDITION {
        printf("n_masking_steps = %d\n", n_masking_steps);
    }
#endif


    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
#ifdef DEBUG_POWER_FWD
        if PRINT_CONDITION {
            printf("power fwd KERNEL: masking_step = %d, m_block = %d, n_block = %d\n", masking_step, m_block, n_block);
        }
#endif

        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        // make sure everything is loaded
        power::cp_async_wait<0>();
        __syncthreads();

#ifdef DEBUG_POWER_FWD
        if PRINT_CONDITION {
            printf("m_block: %d, n_block: %d, masking_step: %d, sK: \n", m_block, n_block, masking_step);
            print_tensor(sK);
            print("\n");
            printf("m_block: %d, n_block: %d, masking_step: %d, sQ: \n", m_block, n_block, masking_step);
            print_tensor(sQ);
            print("\n");
        }
        __syncthreads();
#endif

        // Load V
        // We need to clear OOM elements in V because it's in the reduction axis
        load_V(n_block, /*NO_OOB=*/!is_last_m_block, /*Clear_OOB_MN=*/true);
        cute::cp_async_fence();

        // compute S = QK^T
        power::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        // Apply causal mask
        mask.template apply_mask<Is_causal, Is_even_MN, NormalSpace>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

#ifdef DEBUG_POWER_FWD
        DEBUG_PRINT(acc_s, "before abslogp");
#endif

        // Enter logspace
        Tensor scores = make_tensor(acc_s.data(), power::convert_layout_acc_rowcol(acc_s.layout()));
        Tensor signs = make_tensor<bool>(scores.layout());

        CUTE_UNROLL
        for(int i = 0; i < size(scores); i++) {
            signs(i) = scores(i) >= 0;
        }

        if constexpr (FlashEquivalent || NormalSpace) {
            // S = S / stabilizer
            CUTE_UNROLL
            for (int i = 0; i < size(scores); i++) {
                scores(i) = scores(i) / (NormalSpace ? params.stabilizer_p : params.stabilizer);
            }
        } else {
            // S = p*log(|S| + eps) - log_stabilizer
            if (params.use_multiplier) {
                power::apply_abslogp<true, Deg>(scores, params.ε, params.log_stabilizer);
            } else {
                power::apply_abslogp<false, Deg>(scores, params.ε, params.log_stabilizer);
            }
        }

#ifdef DEBUG_POWER_FWD
        DEBUG_PRINT(acc_s, "after abslogp");
#endif

#ifdef DEBUG_POWER_FWD
        if (PRINT_CONDITION) {
            printf("m_block = %d, n_block = %d, sGQ:\n", m_block, n_block);
            print_tensor(sGQ);
            printf("\n");
            printf("m_block = %d, n_block = %d, sGK:\n", m_block, n_block);
            print_tensor(sGK);
            printf("\n");
        }
#endif
        // Apply gating
        if constexpr (IsGating) {
            // !NormalSpace: S = S + (GQ @ 1^T - 1 @ GK^T)
            // NormalSpace: S = S * (GQ @ 1^T / 1 @ GK^T)^{1/p}
            power::apply_gating</*masked=*/true, /*logspace=*/!NormalSpace, typename Kernel_traits::TiledMma>(acc_s, sGQ, sGK, Deg);
        }

#ifdef DEBUG_POWER_FWD
        DEBUG_PRINT(acc_s, "scoresafter gating");
#endif

        // wait for V
        power::cp_async_wait<0>();
        __syncthreads();

        if (n_block > 0) {
            load_K(n_block - 1, /*NO_OOB=*/true, /*Clear_OOB_MN=*/false);
            if constexpr (IsGating) {
                load_GK(n_block - 1, /*Clear_OOB_MN=*/false);
            }
        }
        cute::cp_async_fence();

        // Take to exponential
        if constexpr (NormalSpace) {
            if (masking_step == 0) {
                sym_power.power_with_rescale</*Is_first=*/true>(scores, acc_o, acc_norm);
            } else {
                sym_power.power_with_rescale</*Is_first=*/false>(scores, acc_o, acc_norm);
            }
        } else {
            if (masking_step == 0) {
                if constexpr (Deg % 2 == 0) {
                    sym_power.exp_with_rescale</*Is_first=*/true>(scores, acc_o, acc_norm);
                } else {
                    sym_power.exp_with_rescale</*Is_first=*/true>(scores, acc_o, acc_norm, signs);
                }
            } else {
                if constexpr (Deg % 2 == 0) {
                    sym_power.exp_with_rescale</*Is_first=*/false>(scores, acc_o, acc_norm);
                } else {
                    sym_power.exp_with_rescale</*Is_first=*/false>(scores, acc_o, acc_norm, signs);
                }
            }
        }
        

#ifdef DEBUG_POWER_FWD
        DEBUG_PRINT(acc_s, "after exp2");
        if (PRINT_CONDITION) {
            printf("m_block = %d, srowmax:\n", m_block);
            print_tensor(sym_power.row_max);
            printf("\n");
        }
#endif

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = power::convert_type<Element>(acc_s);
        if (Return_softmax) {
            cute::copy(rP, tSgS);
            __syncthreads();
            tSgS.data() = tSgS.data() + index_t(-kBlockN);
        }


#ifdef DEBUG_POWER_FWD
        __syncthreads();
        if PRINT_CONDITION {
            printf("m_block: %d, n_block: %d, sV: \n", m_block, n_block);
            print_tensor(sV);
            printf("\n");
        }
        __syncthreads();
#endif


        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), power::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        power::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

#ifdef DEBUG_POWER_FWD
        DEBUG_PRINT(acc_o, "after gemm_rs");
#endif

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= 0) {
#ifdef DEBUG_POWER_FWD
            if (PRINT_CONDITION) {
                printf("n_masking_steps = %d, n_block = %d, break\n", n_masking_steps, n_block);
            }
#endif
            --n_block;
            break;
        }
    }

#ifdef DEBUG_POWER_FWD
    if PRINT_CONDITION {
        printf("after first loop: m_block: %d, n_block: %d\n", m_block, n_block);
    }
#endif

    // These are the iterations where we don't need masking on S
    for (; n_block >= 0; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        // wait for everything to be loaded
        power::cp_async_wait<0>();
        __syncthreads();

        // Load V
        // OOB is still possible if m_block is the last iteration
        load_V(n_block, /*NO_OOB=*/false, /*Clear_OOB_MN=*/true);
        cute::cp_async_fence();
       
        power::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        // Enter logspace
        Tensor scores = make_tensor(acc_s.data(), power::convert_layout_acc_rowcol(acc_s.layout()));
        Tensor signs = make_tensor<bool>(scores.layout());

        CUTE_UNROLL
        for(int i = 0; i < size(scores); i++) {
            signs(i) = scores(i) >= 0;
        }

        if constexpr (FlashEquivalent || NormalSpace) {
            // S = S / stabilizer
            CUTE_UNROLL
            for (int i = 0; i < size(scores); i++) {
                scores(i) = scores(i) / (NormalSpace ? params.stabilizer_p : params.stabilizer);
            }
        } else {
            if (params.use_multiplier) {
                power::apply_abslogp<true, Deg>(scores, params.ε, params.log_stabilizer);
            } else {
                power::apply_abslogp<false, Deg>(scores, params.ε, params.log_stabilizer);
            }
        }

        // Apply gating
        if constexpr (IsGating) {
            power::apply_gating</*masked=*/false, /*logspace=*/!NormalSpace, typename Kernel_traits::TiledMma>(acc_s, sGQ, sGK, Deg);
        }

        // Wait for V, then load K
        power::cp_async_wait<0>();
        __syncthreads();
        if (n_block > 0) {
            // OOB is not possible now
            load_K(n_block - 1, /*NO_OOB=*/true, /*Clear_OOB_MN=*/false);
            if constexpr (IsGating) {
                load_GK(n_block - 1, /*Clear_OOB_MN=*/false);
            }
        }
        cute::cp_async_fence();

        // Take to exponential
        if constexpr (NormalSpace) {
            sym_power.power_with_rescale</*Is_first=*/false>(scores, acc_o, acc_norm);
        } else {
            if constexpr (Deg % 2 == 0) {
                sym_power.exp_with_rescale</*Is_first=*/false>(scores, acc_o, acc_norm);
            } else {
                sym_power.exp_with_rescale</*Is_first=*/false>(scores, acc_o, acc_norm, signs);
            }
        }

#ifdef DEBUG_POWER_FWD
        DEBUG_PRINT(acc_s, "after exp2, second loop");
#endif

        Tensor rP = power::convert_type<Element>(acc_s);
        if (Return_softmax) {
            cute::copy(rP, tSgS);
            __syncthreads();
            tSgS.data() = tSgS.data() + index_t(-kBlockN);
        }

        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), power::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));

#ifdef DEBUG_POWER_FWD
        if PRINT_CONDITION {
            printf("second loop: m_block: %d, n_block: %d, sV: \n", m_block, n_block);
            print_tensor(sV);
            printf("\n");
        }
#endif

        power::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    // Epilogue

    // Sum acc_norm across threads
    SumOp<float> sum_op;
    power_attention::quad_allreduce_(acc_norm, acc_norm, sum_op);

    // first write acc_norm back to shared memory
    __syncthreads(); // necessary since at the last iteration, some threads can still be working with sQ
    Tensor sNorm = make_tensor(reinterpret_cast<ElementAccum *>(&(*sQ.data())), typename Kernel_traits::SmemLayoutNorm{});    // (kBlockM)
    Tensor sRowmax = make_tensor(reinterpret_cast<float *>(&(*sNorm.data()) + int(sNorm.size())), typename Kernel_traits::SmemLayoutRowmax{});
    // because the MMA tiles is 4x1x1 or 8x1x1, we don't need to worry about inter-warp reduction
    for (int mi = 0; mi < size<0>(acc_norm); ++mi) {
        if (tidx % 4 == 0) {
            int idx = (mi / 2) * Kernel_traits::kNWarps * 16 + (tidx / 32) * 16  + (mi % 2) * 8 + (tidx % 32) / 4;
            sNorm(idx) = acc_norm(mi);
            sRowmax(idx) = sym_power.row_max(mi);
        }
    }    
    
    // write back to global memory
    typename Kernel_traits::GmemTiledCopyNorm gmem_tiled_copy_Norm;
    typename Kernel_traits::GmemTiledCopyRowmax gmem_tiled_copy_Rowmax;
    auto gmem_thr_copy_Norm = gmem_tiled_copy_Norm.get_thread_slice(tidx);
    auto gmem_thr_copy_Rowmax = gmem_tiled_copy_Rowmax.get_thread_slice(tidx);
    Tensor tNsNorm = gmem_thr_copy_Norm.partition_S(sNorm);
    Tensor tNgNorm = gmem_thr_copy_Norm.partition_D(gNorm);
    Tensor tRMsRM = gmem_thr_copy_Rowmax.partition_S(sRowmax);
    Tensor tRMgRM = gmem_thr_copy_Rowmax.partition_D(gRowmax);
    __syncthreads();

    Tensor cNorm = make_identity_tensor(make_shape(size<0>(sNorm)));    // (BLK_M) -> (blk_m)
    Tensor tNcNorm = gmem_thr_copy_Norm.partition_S(cNorm);
    // Clear_OOB_MN must be false since we don't want to write zeros to gmem
    // Is_even_MN is false because we always want to check the write in case NThreads < kBlockM
    power::copy1d<false, false, kBlockM>(gmem_tiled_copy_Norm, tNsNorm, tNgNorm, tNcNorm, binfo.actual_seqlen_q - m_block * kBlockM);
    power::copy1d<false, false, kBlockM>(gmem_tiled_copy_Rowmax, tRMsRM, tRMgRM, tNcNorm, binfo.actual_seqlen_q - m_block * kBlockM);

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = power::convert_type<Element>(acc_o);
    Tensor sO = make_tensor(reinterpret_cast<Element *>(&(*sRowmax.data()) + sRowmax.size()), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    const index_t offset_O = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + offset_O), 
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.o_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    power::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    );
}


////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Return_softmax, bool IsGating, bool FlashEquivalent, bool NormalSpace, int Deg, typename Params>
inline __device__ void compute_attn(const Params &params) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;

    power::compute_attn_1rowblock<Kernel_traits, Is_causal, Is_even_MN, Is_even_K, Return_softmax, IsGating, FlashEquivalent, NormalSpace, Deg>(params, bidb, bidh, m_block);
}


////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, int kBlockM, int Log_max_splits, bool Is_even_K, typename Params>
inline __device__ void combine_attn_seqk_parallel(const Params &params) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    constexpr int kMaxSplits = 1 << Log_max_splits;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNThreads = Kernel_traits::kNThreads;

    static_assert(kMaxSplits <= 128, "kMaxSplits must be <= 128");
    static_assert(kBlockM == 4 || kBlockM == 8 || kBlockM == 16 || kBlockM == 32, "kBlockM must be 4, 8, 16 or 32");
    static_assert(kNThreads == 128, "We assume that each block has 128 threads");

    // Shared memory.
    // kBlockM + 1 instead of kBlockM to reduce bank conflicts.
    __shared__ ElementAccum sLSE[kMaxSplits][kBlockM + 1];

    // The thread and block index.
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;

    const index_t lse_size = params.b * params.h * params.seqlen_q;

    const index_t row_offset_lse = bidx * kBlockM;
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lseaccum_ptr) + row_offset_lse),
                                   Shape<Int<kMaxSplits>, Int<kBlockM>>{},
                                   make_stride(lse_size, _1{}));

    // LSE format is different depending on params.unpadded_lse and params.seqlenq_ngroups_swapped, see comment in get_lse_tile.
    // This tensor's layout maps row_offset_lse to {bidb, bidh, q_offset}.
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                              Shape<Int<kBlockM>>{}, Stride<_1>{});

    // This layout maps row_offset_lse to {bidh, q_offset, bidb} or {bidh, bidb, q_offset}.
    Layout flat_layout = make_layout(lse_size);
    Layout orig_layout = make_layout(make_shape(params.seqlen_q, params.h, params.b));
    auto transposed_stride = params.seqlenq_ngroups_swapped ? make_stride(params.b, params.seqlen_q * params.b, 1) : make_stride(1, params.seqlen_q * params.b, params.seqlen_q);
    Layout remapped_layout = make_layout(make_shape(params.seqlen_q, params.h, params.b), transposed_stride);
    Layout final_layout = cute::composition(remapped_layout, cute::composition(orig_layout, flat_layout));

    Tensor gLSE_unpadded = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr)), final_layout);

    constexpr int kNLsePerThread = (kMaxSplits * kBlockM + kNThreads - 1) / kNThreads;

    // Read the LSE values from gmem and store them in shared memory, then transpose them.
    constexpr int kRowsPerLoadLSE = kNThreads / kBlockM;
    #pragma unroll
    for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadLSE + tidx / kBlockM;
        const int col = tidx % kBlockM;
        ElementAccum lse = (row < params.num_splits && col < lse_size - bidx * kBlockM) ? gLSEaccum(row, col) : -INFINITY;
        if (row < kMaxSplits) { sLSE[row][col] = lse; }
        // if (bidx == 0 && tidx < 32) { printf("tidx = %d, row = %d, col = %d, lse = %f\n", tidx, row, col, lse); }
    }
    // if (bidx == 1 && tidx < 32) { printf("tidx = %d, row_offset_lse = %d, lse = %f\n", tidx, row_offset_lse, lse_accum(0)); }
    __syncthreads();
    Tensor lse_accum = make_tensor<ElementAccum>(Shape<Int<kNLsePerThread>>{});
    constexpr int kRowsPerLoadTranspose = std::min(kRowsPerLoadLSE, kMaxSplits);
    // To make sure that kMaxSplits is within 1 warp: we decide how many elements within kMaxSplits
    // each thread should hold. If kMaxSplits = 16, then each thread holds 2 elements (128 threads,
    // kBlockM rows, so each time we load we can load 128 / kBlockM rows).
    // constexpr int kThreadsPerSplit = kMaxSplits / kRowsPerLoadTranspose;
    // static_assert(kThreadsPerSplit <= 32);
    static_assert(kRowsPerLoadTranspose <= 32);
    static_assert(kNLsePerThread * kRowsPerLoadTranspose <= kMaxSplits);
    #pragma unroll
    for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadTranspose + tidx % kRowsPerLoadTranspose;
        const int col = tidx / kRowsPerLoadTranspose;
        lse_accum(l) = (row < kMaxSplits && col < kBlockM) ? sLSE[row][col] : -INFINITY;
        // if (bidx == 0 && tidx < 32) { printf("tidx = %d, row = %d, col = %d, lse = %f\n", tidx, row, col, lse_accum(l)); }
    }

    // Compute the logsumexp of the LSE along the split dimension.
    ElementAccum lse_max = lse_accum(0);
    #pragma unroll
    for (int l = 1; l < kNLsePerThread; ++l) { lse_max = max(lse_max, lse_accum(l)); }
    MaxOp<float> max_op;
    lse_max = Allreduce<kRowsPerLoadTranspose>::run(lse_max, max_op);
    lse_max = lse_max == -INFINITY ? 0.0f : lse_max;  // In case all local LSEs are -inf
    float lse_sum = expf(lse_accum(0) - lse_max);
    #pragma unroll
    for (int l = 1; l < kNLsePerThread; ++l) { lse_sum += expf(lse_accum(l) - lse_max); }
    SumOp<float> sum_op;
    lse_sum = Allreduce<kRowsPerLoadTranspose>::run(lse_sum, sum_op);
    // For the case where all local lse == -INFINITY, we want to set lse_logsum to INFINITY. Otherwise
    // lse_logsum is log(0.0) = -INFINITY and we get NaN when we do lse_accum(l) - lse_logsum.
    ElementAccum lse_logsum = (lse_sum == 0.f || lse_sum != lse_sum) ? INFINITY : logf(lse_sum) + lse_max;
    // if (bidx == 0 && tidx < 32) { printf("tidx = %d, lse = %f, lse_max = %f, lse_logsum = %f\n", tidx, lse_accum(0), lse_max, lse_logsum); }
    if (tidx % kRowsPerLoadTranspose == 0 && tidx / kRowsPerLoadTranspose < kBlockM) {
        if (params.unpadded_lse) {
            const index_t lse_offset = row_offset_lse + tidx / kRowsPerLoadTranspose;
            if (lse_offset < lse_size) {
                gLSE_unpadded(lse_offset) = lse_logsum;
            }
        } else {
            gLSE(tidx / kRowsPerLoadTranspose) = lse_logsum;
        }
    }
    // Store the scales exp(lse - lse_logsum) in shared memory.
    #pragma unroll
    for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadTranspose + tidx % kRowsPerLoadTranspose;
        const int col = tidx / kRowsPerLoadTranspose;
        if (row < params.num_splits && col < kBlockM) { sLSE[row][col] = expf(lse_accum(l) - lse_logsum); }
    }
    __syncthreads();

    const index_t row_offset_oaccum = bidx * kBlockM * params.d_rounded;
    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.oaccum_ptr) + row_offset_oaccum),
                                 Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                 Stride<Int<kHeadDim>, _1>{});
    constexpr int kBlockN = kNThreads / kBlockM;
    using GmemLayoutAtomOaccum = Layout<Shape<Int<kBlockM>, Int<kBlockN>>, Stride<Int<kBlockN>, _1>>;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    GmemTiledCopyOaccum gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_S(gOaccum);
    Tensor tOrO = make_tensor<ElementAccum>(shape(tOgOaccum));
    Tensor tOrOaccum = make_tensor<ElementAccum>(shape(tOgOaccum));
    clear(tOrO);

    // Predicates
    Tensor cOaccum = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    // Repeat the partitioning with identity layouts
    Tensor tOcOaccum = gmem_thr_copy_Oaccum.partition_S(cOaccum);
    Tensor tOpOaccum = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpOaccum); ++k) { tOpOaccum(k) = get<1>(tOcOaccum(0, 0, k)) < params.d; }
    }
    // Load Oaccum in then scale and accumulate to O
    for (int split = 0; split < params.num_splits; ++split) {
        power::copy</*Is_even_MN=*/false, Is_even_K>(
            gmem_tiled_copy_Oaccum, tOgOaccum, tOrOaccum, tOcOaccum, tOpOaccum, params.b * params.h * params.seqlen_q - bidx * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOrOaccum); ++m) {
            int row = get<0>(tOcOaccum(0, m, 0));
            ElementAccum lse_scale = sLSE[split][row];
            #pragma unroll
            for (int k = 0; k < size<2>(tOrOaccum); ++k) {
                #pragma unroll
                for (int i = 0; i < size<0>(tOrOaccum); ++i) {
                    tOrO(i, m, k) += lse_scale * tOrOaccum(i, m, k);
                }
            }
        }
        tOgOaccum.data() = tOgOaccum.data() + params.b * params.h * params.seqlen_q * params.d_rounded;
    }

    Tensor rO = power::convert_type<Element>(tOrO);
    // Write to gO
    #pragma unroll
    for (int m = 0; m < size<1>(rO); ++m) {
        const int idx = bidx * kBlockM + get<0>(tOcOaccum(0, m, 0));
        if (idx < params.b * params.h * params.seqlen_q) {
            const int batch_idx = idx / (params.h * params.seqlen_q);
            const int head_idx = (idx - batch_idx * (params.h * params.seqlen_q)) / params.seqlen_q;
            // The index to the rows of Q
            const int row = idx - batch_idx * (params.h * params.seqlen_q) - head_idx * params.seqlen_q;
            auto o_ptr = reinterpret_cast<Element *>(params.o_ptr) + batch_idx * params.o_batch_stride
                + head_idx * params.o_head_stride + row * params.o_row_stride;
            #pragma unroll
            for (int k = 0; k < size<2>(rO); ++k) {
                if (Is_even_K || tOpOaccum(k)) {
                    const int col = get<1>(tOcOaccum(0, m, k));
                    Tensor gO = make_tensor(make_gmem_ptr(o_ptr + col),
                                            Shape<Int<decltype(size<0>(rO))::value>>{}, Stride<_1>{});
                    // TODO: Should check if this is using vectorized store, but it seems pretty fast
                    copy(rO(_, m, k), gO);
                    // if (bidx == 0 && tidx == 0) { printf("tidx = %d, idx = %d, batch_idx = %d, head_idx = %d, row = %d, col = %d\n", tidx, idx, batch_idx, head_idx, row, col); print(rO(_, m, k)); print(gO); }
                    // reinterpret_cast<uint64_t *>(o_ptr)[col / 4] = recast<uint64_t>(rO)(0, m, k);
                }
            }
        }
    }
}

} // namespace power
