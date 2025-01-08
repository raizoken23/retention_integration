#pragma once
#include <tuple>
#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "state.h"
#include "kernel_traits.h"
#include "utils.h"
#include "sympow.h"
#include "attention/power_utils.h"

// #define CHUNK_STATE_FWD_DEBUG 1
#define DEBUG_THREAD (tid == 0 && dim_id == 0 && head_id == 0 && batch_id == 0 && chunk_id == 0)

namespace state_kernel
{
    using namespace cute;

#define ASSERT_SIZE_EQ(S, D)                \
    CUTE_STATIC_ASSERT(size(S) == size(D)); \
    CUTE_STATIC_ASSERT(rank(S) == rank(D))

    template <typename Kernel_traits>
    __global__ void __launch_bounds__(Kernel_traits::NWarps *cutlass::NumThreadsPerWarp, 1) chunk_state_kernel_fwd(__grid_constant__ const Chunk_state_params params)
    {
        using Element = typename Kernel_traits::Element;
        using ElementAccum = typename Kernel_traits::ElementAccum;
        using index_t = typename Kernel_traits::index_t;
        // Shape parameters
        constexpr int BlockD = Kernel_traits::BlockD;
        constexpr int BlockK = Kernel_traits::BlockK;
        constexpr int Headdim = Kernel_traits::Headdim;
        constexpr int Deg = Kernel_traits::Deg;
        constexpr bool DoubleBuffer = Kernel_traits::DoubleBuffer;
        constexpr int PaddedExpandedDim = Kernel_traits::PaddedExpandedDim;
        constexpr int InnerBlock = Kernel_traits::InnerBlock;
        constexpr int NumOuterBlocks = Headdim / Kernel_traits::OuterBlock;
        constexpr int NumInnerBlocks = Headdim / Kernel_traits::InnerBlock;

        // Shared memory.
        __align__(128) extern __shared__ char smem_[];

        // Thread index
        const int tid = threadIdx.x;

        // Block index
        const int dim_id = blockIdx.x;
        const auto info = state_kernel::binfo(dim_id, Kernel_traits::OuterBlock, Kernel_traits::InnerBlock);
        const auto inner_bid = std::get<0>(info);
        const auto outer_bid = std::get<1>(info);
        const auto is_on_diagonal = std::get<2>(info);

        const int head_id = blockIdx.y;
        const int batch_id = blockIdx.z / params.num_chunks;
        const int chunk_id = blockIdx.z % params.num_chunks;
        int kblock = (params.chunk_seq_len + BlockK - 1) / BlockK - 1;

        // Offset calculators
        const index_t kv_offset = batch_id * params.batch_stride + chunk_id * params.chunk_stride + kblock * BlockK * params.seq_stride + head_id * params.head_stride;
        const index_t s_offset = batch_id * params.batch_stride_s + chunk_id * params.chunk_stride_s + head_id * params.head_stride_s + dim_id * BlockD * Headdim;
        const index_t phi_offset = batch_id * params.batch_stride_phi + chunk_id * params.chunk_stride_phi + kblock * BlockK * params.seq_stride_phi + head_id * params.head_stride_phi + dim_id * BlockD;
        const index_t D_offset = dim_id * Deg;
        const index_t C_offset = dim_id * BlockD;

        // ===================== Global memory tensors =====================
        // Represent the full tensors for K
        Tensor gK = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + kv_offset),
            Shape<Int<BlockK>, Int<Headdim>>{},
            make_stride(params.seq_stride, _1{})); // (BlockK, Headdim)

        // Represent the full tensors for V
        Tensor gV = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + kv_offset),
            Shape<Int<BlockK>, Int<Headdim>>{},
            make_stride(params.seq_stride, _1{})); // (BlockK, Headdim)

        // Represent the full tensors for O
        Tensor gO = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + s_offset),
            Shape<Int<BlockD>, Int<Headdim>>{},
            make_stride(Int<Headdim>{}, _1{}));     // (BlockD, Headdim)

        // Represent the full tensor for PhiK
        Tensor gPhi = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.phi_ptr) + phi_offset),
            Shape<Int<BlockK>, Int<BlockD>>{},
            make_stride(params.seq_stride_phi, _1{})); // (BlockK, BlockD)

        // ===================== Shared memory tensors =====================
        Tensor sK = make_tensor(
            make_smem_ptr(reinterpret_cast<Element *>(smem_)),
            typename Kernel_traits::SmemLayoutK{}); // (BlockK, Headdim)

        Tensor sKt = make_tensor(
            sK.data(),
            typename Kernel_traits::SmemLayoutKt{}); // (Headdim, BlockK)

        Tensor sV = make_tensor(
            (sK.data() + (DoubleBuffer ? 2 : 1) * size(sK)),
            typename Kernel_traits::SmemLayoutV{}); // (BlockK, Headdim)

        Tensor sVt = make_tensor(
            sV.data(),
            typename Kernel_traits::SmemLayoutVt{}); // (Headdim, BlockK)

        Tensor sVtNoSwizzle = make_tensor(
            sV.data(),
            typename Kernel_traits::SmemLayoutVtNoSwizzle{});

        Tensor sPK = make_tensor(
            make_smem_ptr(reinterpret_cast<Element *>(&(*sV.data()) + size(sV))),
            typename Kernel_traits::SmemLayoutPhiK{}); // (BlockK, BlockD)

        Tensor sPKt = make_tensor(
            sPK.data(),
            typename Kernel_traits::SmemLayoutPhiKt{}); // (BlockD, BlockK)

        // be careful we are reusing smem
        Tensor sO = make_tensor(
            make_smem_ptr(reinterpret_cast<Element *>(smem_)),
            typename Kernel_traits::SmemLayoutO{}); // (BlockD, Headdim)

        typename Kernel_traits::GmemCopyTileK gmem_tiled_copy_K;
        typename Kernel_traits::GmemCopyTileV gmem_tiled_copy_V;
        typename Kernel_traits::GmemCopyTilePhiK gmem_tiled_copy_PhiK;
        auto gmem_thr_copy_K = gmem_tiled_copy_K.get_thread_slice(tid);
        auto gmem_thr_copy_V = gmem_tiled_copy_V.get_thread_slice(tid);
        auto gmem_thr_copy_PhiK = gmem_tiled_copy_PhiK.get_thread_slice(tid);

        Tensor tKgK = gmem_thr_copy_K.partition_S(gK);           // ((CPY_ATOM_I, CPY_ATOM_J), TILE_M, TILE_N)
        Tensor tKsK = gmem_thr_copy_K.partition_D(sK);           // ((CPY_ATOM_I, CPY_ATOM_J), TILE_M, TILE_N)
        Tensor tVgV = gmem_thr_copy_V.partition_S(gV);           // ((CPY_ATOM_I, CPY_ATOM_J), TILE_M, TILE_N)
        Tensor tVsV = gmem_thr_copy_V.partition_D(sV);           // ((CPY_ATOM_I, CPY_ATOM_J), TILE_M, TILE_N)
        Tensor tPgP = gmem_thr_copy_PhiK.partition_D(gPhi);      // ((CPY_ATOM_I, CPY_ATOM_J), TILE_M, TILE_N)
        Tensor tPsP = gmem_thr_copy_PhiK.partition_S(sPK);       // ((CPY_ATOM_I, CPY_ATOM_J), TILE_M, TILE_N)

        // predicates for loading stuff
        Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
        Tensor tKVcKV = gmem_thr_copy_K.partition_S(cKV);
        Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

        // Loaders
        auto load_K = [&]() {
            state_kernel::copy</*Is_even_MN=*/false>(gmem_tiled_copy_K, tKgK, tKsK, tKVcKV, std::min(BlockK, params.chunk_seq_len - kblock * BlockK));
            tKgK.data() = tKgK.data() + index_t(-BlockK * params.seq_stride);
        };
        auto load_V = [&]() {
            state_kernel::copy</*Is_even_MN=*/false>(gmem_tiled_copy_V, tVgV, tVsV, tKVcKV, std::min(BlockK, params.chunk_seq_len - kblock * BlockK));
            tVgV.data() = tVgV.data() + index_t(-BlockK * params.seq_stride);
        };
        auto save_Phi = [&]() {
            Tensor cP = make_identity_tensor(gPhi.shape());
            Tensor tPcP = gmem_thr_copy_PhiK.partition_S(cP);

            state_kernel::copy</*Is_even_MN=*/false>(gmem_tiled_copy_PhiK, tPsP, tPgP, tPcP, std::min(BlockK, params.chunk_seq_len - kblock * BlockK));
            tPgP.data() = tPgP.data() + index_t(-BlockK * params.seq_stride_phi);
        };

        // MMA stuff
        typename Kernel_traits::TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_thread_slice(tid);

        // These handles smem->register copy for mma
        auto smem_tiled_copy_Vt = make_tiled_copy_B(
            typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
        auto smem_thr_copy_Vt = smem_tiled_copy_Vt.get_thread_slice(tid);
        Tensor tSsVt = smem_thr_copy_Vt.partition_S(sVt); // ((2, 2), V_feature/8)

        Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<BlockD>, Int<Headdim>>{});
        using regA_layout = decltype(thr_mma.partition_fragment_A(sPKt).layout());
        using regA_rowcol = decltype(state_kernel::convert_layout_rA_rowcol(regA_layout{}));

        state_kernel::SymowStateExpanderBlockP2<Kernel_traits> sympow;

        // main steps:
        // 1. copy D and C to shared memory
        // 2. copy K into shared memory
        // 3. If double buffer, start async copy of next K
        // 4. If not KV_shared_mem_, start async copy of V
        // 5. wait for K to be in smem. If KV_shared_mem_, syncthreads and start copy V
        // 6. each thread load its corresponding D, C, use D to locate K, expand the state in register
        // 7. accumualte norm on register
        // 8. do mma_rs between phi(K)^T and V to get S, accumulate

        // Prologue
        // clear(acc_n);
        clear(acc_o);

        // Initialize position for sK, if double buffer
        if (DoubleBuffer && kblock % 2 == 1)
        {
            tKsK.data() = tKsK.data() + int(size(sK));
            sKt.data() = sKt.data() + int(size(sK));
        }

        // start loading K
        load_K();
        if (DoubleBuffer) {
            tKsK.data() = tKsK.data() + int((kblock % 2 == 0) ? size(sK) : -size(sK));
        }
        cute::cp_async_fence();

        // Main loop, reverse order to save 1 register and index calculation
        for (; kblock >= 0; --kblock)
        {
            // start loading V
            load_V();
            cute::cp_async_fence();

            // start loading next K
            if (DoubleBuffer && kblock > 0)
            {
                load_K();
                tKsK.data() = tKsK.data() + int((kblock % 2 == 1) ? size(sK) : -size(sK));
            }
            cute::cp_async_fence();

            // wait for K to be in smem
            cute::cp_async_wait<2>();
            __syncthreads();

#ifdef CHUNK_STATE_FWD_DEBUG
            __syncthreads();
            if (DEBUG_THREAD) {
                printf("kblock: %d, outer_bid: %d, inner_bid: %d, is_on_diagonal: %d, sK: \n", kblock, outer_bid, inner_bid, is_on_diagonal);
                Tensor sK_now = make_tensor(sK.data() + (DoubleBuffer ? (kblock % 2 == 0 ? 0 : size(sK)) : 0), sK.layout());
                print_tensor(sK_now);
                printf("\n");
                printf("acc_n: \n");
                for (int i = 0; i < size(acc_n); i++) {
                    printf("%f ", acc_n[i]);
                }
                printf("\n");
            }
            __syncthreads();
#endif

            // expand states
            Tensor tSrPKt = thr_mma.partition_fragment_A(sPKt);
            if (params.expand) {
                tSrPKt = BINARY_DIM_SWITCH(inner_bid, INNER_BID, Headdim/InnerBlock, [&]() {
                    return sympow.expandState<INNER_BID>(sKt, sPKt, tiled_mma, outer_bid, is_on_diagonal); // ((2, 2, 2), TILE_D, TILE_T)
                });
            }

#ifdef CHUNK_STATE_FWD_DEBUG
            Tensor tSsPKt_ = thr_mma.partition_A(sPKt);
            cute::copy(tSrPKt, tSsPKt_);
            __syncthreads();
            if (DEBUG_THREAD) {
                printf("kblock: %d, sPK: \n", kblock);
                print_tensor(sPK);
                printf("\n");
            }
            __syncthreads();
#endif
            
            if constexpr (DoubleBuffer) {
                sKt.data() = sKt.data() + int(((kblock) % 2 == 0) ? size(sK) : -size(sK));
            }

            // optionally save Phi(K)^T to smem
            if (params.return_phi) {
                Tensor tSsPKt = thr_mma.partition_A(sPKt);
                cute::copy(tSrPKt, tSsPKt);
                __syncthreads();
            }

            // make sure V is in smem
            cute::cp_async_wait<1>();
            __syncthreads();

#ifdef CHUNK_STATE_FWD_DEBUG
            __syncthreads();
            if (DEBUG_THREAD) {
                printf("kblock: %d, sV: \n", kblock);
                print_tensor(sV);
                printf("\n");
            }
            __syncthreads();
#endif

            // do mma between phi(K)^T and V
            Tensor tSrVt = thr_mma.partition_fragment_B(sVtNoSwizzle); // ((2, 2), V_feature/8)
            state_kernel::gemm_rs(acc_o, tSrPKt, tSrVt, tSsVt, tiled_mma, smem_tiled_copy_Vt, smem_thr_copy_Vt);
            __syncthreads();

            // if not double buffer, this is where we load next K
            if (!DoubleBuffer && kblock > 0) {
                load_K();
            }
            cute::cp_async_fence();

            // save Phi if return_phi is true
            if (params.return_phi)
            {
                save_Phi();
                __syncthreads();
            }
        }

        // Epilogue
        // First reduce acc_n within warps
        __syncthreads();

        // copy acc_o to smem first
        // TODO (sean): use stmatrix here
        Tensor rO = state_kernel::convert_type<Element>(acc_o);
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));
        auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
        auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tid);
        Tensor tOcsO = smem_thr_copy_O.partition_S(cO);
        Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
        Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

        state_kernel::copy</*Is_even_MN=*/true>(smem_tiled_copy_O, taccOrO, taccOsO, tOcsO, BlockD);
        __syncthreads();

#ifdef CHUNK_STATE_FWD_DEBUG
        __syncthreads();
        if (DEBUG_THREAD) {
            printf("kblock: %d, sO: \n", kblock);
            print_tensor(sO);
            printf("\n");
        }
        __syncthreads();
#endif

        // copy back to global memory
        typename Kernel_traits::GmemCopyTileO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tid);
        Tensor tOcO = gmem_thr_copy_O.partition_S(cO);

        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOsO = gmem_thr_copy_O.partition_S(sO);

        state_kernel::copy</*Is_even_MN=*/false>(gmem_tiled_copy_O, tOsO, tOgO, tOcO, std::min(BlockD, PaddedExpandedDim - dim_id * BlockD));
    }

} // namespace state_kernel
