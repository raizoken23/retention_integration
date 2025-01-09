/***************************************************************************************************
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
#include "gating.h"

// #define DEBUG_POWER_BWD_DKDV 1
// #define DEBUG_POWER_BWD_DQ 1
// #define DEBUG_POWER_BWD_DKDVDQ 1
#define DEBUGGER_THREAD (tidx == 0 && n_block == 0 && bidh == 0 && bidb == 0)

namespace power {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

// template<typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool IsGating, typename Params>
// inline __device__ void compute_dk_dv_1colblock(const Params &params, const int bidb, const int bidh, const int n_block) {

//     // Math:
//     // Forward:
//     // S = QK^T
//     // M = Causal Mask
//     // T = p * log(abs(S) + ε)
//     // Z = T + p * (G_Q @ 1^T - 1 @ G_K^T)
//     // P = exp(Z * M)
//     // Y = P @ V
//     // y = P @ 1

//     // Backward:
//     // dV = P^T @ dY
//     // dP = dY @ V + dy @ 1^T
//     // dZ = dP * P
//     // dT = dZ
//     // dS = dT * p * sign(S) / (abs(S) + ε)
//     // dK = dS^T @ Q
//     // dQ = dS @ K
//     // dG_Q = (dZ * p) @ 1
//     // dG_K = - (dZ * p)^T @ 1

//     // Options:
//     // Is_V_in_regs: put V in registers instead of smem, this reduces smem pressure at the cost of regitser pressure

//     // Main steps (dK dV)
//     // Prologue:
//     //     * [if Is_V_in_regs] start loading V
//     //     * start loading K
//     //     * [if not Is_V_in_regs] start loading V
//     //     * start loading G_K
//     //     * start loading Q[0], G_Q[0]
//     //     * start loading dY[0], dy[0]
//     //     * if Is_V_in_regs, wait for V, put V in registers
//     //     * init acc_dV, acc_dK, acc_dGK
//     // Main loop: for each iteration i
//     //     * [if double buffer && i < max] start loading Q[i+1]
//     //     * compute S[i] = Q[i] @ K^T, save S[i] copy
//     //     * compute T[i] = p * log(abs(S[i]) + ε)
//     //     * compute Z[i] = T[i] + p * (G_Q[i] @ 1^T - 1 @ G_K^T)
//     //     * start loading G_Q[i+1]
//     //     * compute P[i] = exp(Z[i] * M)
//     //     * put P[i] in smem
//     //     * compute acc_dV += P[i]^T @ dY[i]
//     //     * compute dP[i] = dY[i] @ V
//     //     * start loading dY[i+1]
//     //     * compute dP[i] = dP[i] + dy @ 1^T
//     //     * start loading dy[i+1]
//     //     * compute dZ[i] = dT[i] = dP[i] * exp(Z[i] * M) # use Z[i] registers for dZ[i]
//     //     * compute dS[i] = dT[i] * p * sign(S[i]) / (abs(S[i]) + ε) # use S[i] registers for dS[i]
//     //     * compute acc_dGK += -(dZ[i] * p)^T @ 1
//     //     * put dS[i] in smem # optionally use sV
//     //     * compute acc_dK += dS[i]^T @ Q[i] # this is why Q needs double buffering, it's required both in the beginning and so late in the loop
//     //     * start loading Q[i+1]
//     // Epilogue:
//     //     * reduce acc_dGK inter warp and intra warp, put in smem, write to gmem
//     //     * put acc_dK in smem, write to gmem
//     //     * put acc_dV in smem, write to gmem

//     using Element = typename Kernel_traits::Element;
//     using ElementAccum = typename Kernel_traits::ElementAccum;
//     using index_t = typename Kernel_traits::index_t;

//     // Shared memory.
//     extern __shared__ char smem_[];

//     // The thread index.
//     const int tidx = threadIdx.x;

//     constexpr int kBlockM = Kernel_traits::kBlockM;
//     constexpr int kBlockN = Kernel_traits::kBlockN;
//     constexpr int kHeadDim = Kernel_traits::kHeadDim;
//     constexpr int MMA_N_SdP = kBlockN / decltype(typename Kernel_traits::TiledMmaSdP{}.template tile_size_mnk<1>())::value; /*Number of tiles along the N dimension*/
//     constexpr int AtomLayoutMS = Kernel_traits::AtomLayoutMSdP;
//     constexpr int kNWarps = Kernel_traits::kNWarps;
//     constexpr int AtomLayoutNS = kNWarps / AtomLayoutMS;
//     constexpr bool Double_buffer = !Kernel_traits::No_double_buffer;
//     constexpr int kNThreads = Kernel_traits::kNThreads;
//     constexpr bool is_bf16 = std::is_same_v<Element, cutlass::bfloat16_t>;

//     const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
//     if (n_block * kBlockN >= binfo.actual_seqlen_k) return;

//     int m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM);
//     bool is_last_n_block = n_block == cute::ceil_div(binfo.actual_seqlen_k, kBlockN) - 1;

//     const index_t offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
//         + (m_block_max - 1) * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
//     const index_t offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
//         + n_block * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
//     const index_t offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
//         + n_block * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
//     const index_t offset_gq = binfo.q_offset(params.g_batch_stride, params.g_row_stride, bidb)
//         + (m_block_max - 1) * kBlockM * params.g_row_stride + static_cast<index_t>(bidh);
//     const index_t offset_gk = binfo.k_offset(params.g_batch_stride, params.g_row_stride, bidb)
//         + n_block * kBlockN * params.g_row_stride + static_cast<index_t>(bidh); // we want bidh instead of bidh / params.h_h_k_ratio for special handling for GQA
//     const index_t offset_dY = binfo.q_offset(params.dY_batch_stride, params.dY_row_stride, bidb)
//         + (m_block_max - 1) * kBlockM * params.dY_row_stride + bidh * params.dY_head_stride;
//     const index_t offset_dy = binfo.q_offset(params.dy_batch_stride, params.dy_row_stride, bidb)
//         + (m_block_max - 1) * kBlockM * params.dy_row_stride + bidh * params.dy_head_stride;

//     ////////////////Global Memory////////////////////
//     Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + offset_q),
//                             Shape<Int<kBlockM>, Int<kHeadDim>>{},
//                             make_stride(params.q_row_stride, _1{}));
//     Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + offset_k),
//                             Shape<Int<kBlockN>, Int<kHeadDim>>{},
//                             make_stride(params.k_row_stride, _1{}));
//     Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + offset_v),
//                             Shape<Int<kBlockN>, Int<kHeadDim>>{},
//                             make_stride(params.v_row_stride, _1{}));
//     // gGQ won't be used if IsGating is false
//     Tensor gGQ = make_tensor(make_gmem_ptr(reinterpret_cast<float *>(IsGating ? params.log_g_q_ptr : params.q_ptr) + offset_gq), 
//                             Shape<Int<kBlockM>>{},
//                             make_stride(params.g_row_stride));
//     // We could use the same gating factor for Q and K here to increase L2 cache hits
//     // but that's only valid for training
//     Tensor gGK = make_tensor(make_gmem_ptr(reinterpret_cast<float *>(IsGating ? params.log_g_k_ptr : params.k_ptr) + offset_gk),
//                              Shape<Int<kBlockN>>{},
//                              make_stride(params.g_row_stride));
//     Tensor gdY = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dY_ptr) + offset_dY),
//                              Shape<Int<kBlockM>, Int<kHeadDim>>{},
//                              make_stride(params.dY_row_stride, _1{}));
//     Tensor gdy = make_tensor(make_gmem_ptr(reinterpret_cast<float *>(params.dy_ptr) + offset_dy),
//                             Shape<Int<kBlockM>>{},
//                             make_stride(params.dy_row_stride));
//     Tensor gdGK = make_tensor(make_gmem_ptr(reinterpret_cast<float *>(IsGating ? params.dlog_g_k_ptr : params.k_ptr) + offset_gk),
//                              Shape<Int<kBlockN>>{},
//                              make_stride(params.g_row_stride));
//     ////////////////Shared Memory////////////////////
//     Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
//                             typename Kernel_traits::SmemLayoutQdY{});
//     Tensor sQt = make_tensor(sQ.data(),
//                             typename Kernel_traits::SmemLayoutQdYtransposed{});
//     Tensor sQtNoSwizzle = make_tensor(sQ.data(),
//                             typename Kernel_traits::SmemLayoutQdYtransposedNoSwizzle{});
//     // Double buffer for sQ
//     Tensor sdY = make_tensor(sQ.data() + (Double_buffer ? 2 : 1) * size(sQ),
//                             typename Kernel_traits::SmemLayoutQdY{});
//     Tensor sdYt = make_tensor(sdY.data(),
//                              typename Kernel_traits::SmemLayoutQdYtransposed{});
//     Tensor sdYtransposedNoSwizzle = make_tensor(sdY.data(),
//                                                 typename Kernel_traits::SmemLayoutQdYtransposedNoSwizzle{});
//     Tensor sdy = make_tensor(make_smem_ptr(reinterpret_cast<float *>(&(*sdY.data()) + size(sdY))),
//                             typename Kernel_traits::SmemLayoutdy{});
//     Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(&(*sdy.data()) + size(sdy))),
//                             typename Kernel_traits::SmemLayoutKV{});
//     Tensor sKt = make_tensor(sK.data(),
//                             typename Kernel_traits::SmemLayoutKtransposed{});
//     Tensor sKtNoSwizzle = make_tensor(sK.data(),
//                             typename Kernel_traits::SmemLayoutKtransposedNoSwizzle{});
//     Tensor sV = make_tensor(sK.data() + size(sK),
//                             typename Kernel_traits::SmemLayoutKV{});
//     Tensor sdS = make_tensor(!Kernel_traits::Is_V_in_regs ? sV.data() + size(sV) : sK.data() + size(sK),
//                              typename Kernel_traits::SmemLayoutPdS{});
//     Tensor sdSt = make_tensor(sdS.data(),
//                             typename Kernel_traits::SmemLayoutPdStransposed{});
//     Tensor sdStNoSwizzle = make_tensor(sdS.data(),
//                             typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});
//     Tensor sP = make_tensor(sdS.data() + size(sdS),
//                             typename Kernel_traits::SmemLayoutPdS{});
//     Tensor sPt = make_tensor(sP.data(),
//                             typename Kernel_traits::SmemLayoutPdStransposed{});
//     Tensor sPtNoSwizzle = make_tensor(sP.data(),
//                             typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});
//     Tensor sGQ = make_tensor(make_smem_ptr(reinterpret_cast<float *>(&(*sP.data()) + (IsGating ? size(sP) : 0))),
//                              typename Kernel_traits::SmemLayoutGQ{});
//     Tensor sGK = make_tensor(make_smem_ptr(reinterpret_cast<float *>(&(*sGQ.data()) + (IsGating ? size(sGQ) : 0))),
//                              typename Kernel_traits::SmemLayoutGK{});
//     Tensor sdGK = make_tensor(sGK.data(), typename Kernel_traits::SmemLayoutGK{});

//     typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
//     auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
//     typename Kernel_traits::GmemTiledCopydy gmem_tiled_copy_dy;
//     auto gmem_thr_copy_dy = gmem_tiled_copy_dy.get_thread_slice(tidx);
//     typename Kernel_traits::GmemTiledCopyLogG gmem_tiled_copy_LogG;
//     auto gmem_thr_copy_LogG = gmem_tiled_copy_LogG.get_thread_slice(tidx);
//     typename Kernel_traits::GmemTiledCopydLogG gmem_tiled_copy_dLogG;
//     auto gmem_thr_copy_dLogG = gmem_tiled_copy_dLogG.get_thread_slice(tidx);

//     Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
//     Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
//     Tensor tdYgdY = gmem_thr_copy_QKV.partition_S(gdY);
//     Tensor tdYsdY = gmem_thr_copy_QKV.partition_D(sdY);
//     Tensor tdygdy = gmem_thr_copy_dy.partition_S(gdy);
//     Tensor tdysdy = gmem_thr_copy_dy.partition_D(sdy);
//     Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
//     Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
//     Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
//     Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
//     Tensor tGQgGQ = gmem_thr_copy_LogG.partition_S(gGQ);  // (GCPY, GCPY_M)
//     Tensor tGQsGQ = gmem_thr_copy_LogG.partition_D(sGQ);
//     Tensor tGKgGK = gmem_thr_copy_LogG.partition_S(gGK);  // (GCPY, GCPY_N)
//     Tensor tGKsGK = gmem_thr_copy_LogG.partition_D(sGK);

//     typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
//     auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
//     Tensor tSrQ = thr_mma_sdp.partition_fragment_A(sQ);         // (MMA,MMA_N,MMA_K)
//     Tensor tSrK = thr_mma_sdp.partition_fragment_B(sK);         // (MMA,MMA_N,MMA_K)
//     Tensor tdPrdY = thr_mma_sdp.partition_fragment_A(sdY);      // (MMA,MMA_N,MMA_K)
//     Tensor tdPrV = thr_mma_sdp.partition_fragment_B(sV);        // (MMA,MMA_N,MMA_K)

//     typename Kernel_traits::TiledMmadKV tiled_mma_dkv;
//     auto thr_mma_dkv = tiled_mma_dkv.get_thread_slice(tidx);
//     Tensor tdKrdSt = thr_mma_dkv.partition_fragment_A(sdStNoSwizzle); // (MMA, MMA_N, MMA_N)
//     Tensor tdKrQt = thr_mma_dkv.partition_fragment_B(sQtNoSwizzle);   // (MMA, MMA_K, MMA_N)
//     Tensor tdVrPt = thr_mma_dkv.partition_fragment_A(sPtNoSwizzle);   // (MMA, MMA_N, MMA_N)
//     Tensor tdVrdY = thr_mma_dkv.partition_fragment_B(sdYtransposedNoSwizzle); // (MMA, MMA_K, MMA_N)

//     Tensor acc_dk = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});  // MMA, MMA_N, MMA_K
//     Tensor acc_dv = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});  // MMA, MMA_N, MMA_K

//     //
//     // Copy Atom retiling
//     //

//     auto smem_tiled_copy_QdY = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
//     auto smem_thr_copy_QdY = smem_tiled_copy_QdY.get_thread_slice(tidx);
//     Tensor tSsQ = smem_thr_copy_QdY.partition_S(sQ);
//     Tensor tdPsdY = smem_thr_copy_QdY.partition_S(sdY);

//     // Rearrange dimension N to reduce write bank conflicts
//     auto smem_tiled_copy_KV = power::make_tiled_copy_B_warpcontiguousN<MMA_N_SdP>(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
//     auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);
//     Tensor tSsK = smem_thr_copy_KV.partition_S(sK);
//     Tensor tdPsV = smem_thr_copy_KV.partition_S(sV);

//     // Partition sP and sdS to match the accumulator partitioning
//     // This reduces bank conflicts when writing to sP (but not sdS)
//     auto smem_tiled_copy_PdS = power::make_tiled_copy_C_warpcontiguousN<MMA_N_SdP>(typename Kernel_traits::SmemCopyAtomPdS{}, tiled_mma_sdp);
//     auto smem_thr_copy_PdS = smem_tiled_copy_PdS.get_thread_slice(tidx);
//     Tensor tPsP = smem_thr_copy_PdS.partition_D(sP);      // ((Atom,AtomNum),PIPE_M,PIPE_N)
//     Tensor tdSsdS = smem_thr_copy_PdS.partition_D(sdS);   // ((Atom,AtomNum),PIPE_M,PIPE_N)

//     auto smem_tiled_copy_PdSt = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dkv);
//     auto smem_thr_copy_PdSt = smem_tiled_copy_PdSt.get_thread_slice(tidx);
//     Tensor tdVsPt = smem_thr_copy_PdSt.partition_S(sPt);
//     Tensor tdKsdSt = smem_thr_copy_PdSt.partition_S(sdSt);

//     auto smem_tiled_copy_QdYt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dkv);
//     auto smem_thr_copy_QdYt = smem_tiled_copy_QdYt.get_thread_slice(tidx);
//     Tensor tdVsdYt = smem_thr_copy_QdYt.partition_S(sdYt);
//     Tensor tdKsQt = smem_thr_copy_QdYt.partition_S(sQt);


//     int m_block = m_block_max - 1;
//     int m_block_min = (!Is_causal)
//         ? 0
//         : std::max(0, (n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right) / kBlockM); // To account for unequal seqlen

//     // Predicate tensors for QdY
//     static_assert(size<0>(sQ) == size<0>(sdY), "sQ and sdY must have the same number of rows");
//     static_assert(size<1>(sQ) == size<1>(sdY), "sQ and sdY must have the same number of columns");
//     Tensor cQdY = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M, BLK_K) -> (blk_m, blk_k)
//     Tensor tQdYcQdY = gmem_thr_copy_QKV.partition_D(cQdY);
//     // Allocate predicate tensors for k
//     Tensor tQdYpQdY = make_tensor<bool>(make_shape(size<2>(tSsQ)));
//     if constexpr (!Is_even_K) {
//         #pragma unroll
//         for (int k = 0; k < size(tQdYpQdY); ++k) { tQdYpQdY(k) = get<1>(tQdYcQdY(0, 0, k)) < params.d; }
//     }

//     // Predicate tensors for KV
//     static_assert(size<0>(sV) == size<0>(sK), "sV and sK must have the same number of rows");
//     static_assert(size<1>(sV) == size<1>(sK), "sV and sK must have the same number of columns");
//     Tensor cKV = make_identity_tensor(make_shape(size<0>(sV), size<1>(sV)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
//     Tensor tKVcKV = gmem_thr_copy_QKV.partition_D(cKV);
//     // Allocate predicate tensors for k
//     Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tVsV)));
//     if constexpr(!Is_even_K) {
//         #pragma unroll
//         for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
//     }

//     // Loaders and bumpers, we can remove the if else with templated lambdas
//     // when libtorch is compatible with c++20
//     auto load_GQ = [&](const int m_block, const bool Is_even=true, const bool Clear_OOB_MN=true) {
//         if constexpr (!IsGating) {
//             return;
//         }
//         Tensor cGQ = make_identity_tensor(make_shape(size<0>(sGQ)));    // (BLK_M) -> (blk_m)
//         Tensor tGQcGQ = gmem_thr_copy_LogG.partition_D(cGQ);
//         if (Clear_OOB_MN) {
//             power::copy1d<false, true, kBlockM>(gmem_tiled_copy_LogG, tGQgGQ, tGQsGQ, tGQcGQ, std::min(binfo.actual_seqlen_q - m_block * kBlockM, kBlockM));
//         } else {
//             power::copy1d<false, false, kBlockM>(gmem_tiled_copy_LogG, tGQgGQ, tGQsGQ, tGQcGQ, std::min(binfo.actual_seqlen_q - m_block * kBlockM, kBlockM));
//         }
//     };
//     auto bump_GQ = [&]() {
//         if constexpr (!IsGating) {
//             return;
//         }
//         tGQgGQ.data() = tGQgGQ.data() + -index_t(-kBlockM * params.g_row_stride);
//     };
//     auto load_GK = [&](bool Is_even=true, bool Clear_OOB_MN=true) {
//         if constexpr (!IsGating) {
//             return;
//         }
//         Tensor cGK = make_identity_tensor(make_shape(size<0>(sGK)));    // (BLK_N) -> (blk_n)
//         Tensor tGKcGK = gmem_thr_copy_LogG.partition_D(cGK);
//         if (Clear_OOB_MN) {
//             power::copy1d<false, true, kBlockN>(gmem_tiled_copy_LogG, tGKgGK, tGKsGK, tGKcGK, std::min(binfo.actual_seqlen_k - n_block * kBlockN, kBlockN));
//         } else {
//             power::copy1d<false, false, kBlockN>(gmem_tiled_copy_LogG, tGKgGK, tGKsGK, tGKcGK, std::min(binfo.actual_seqlen_k - n_block * kBlockN, kBlockN));
//         }
//     };
//     auto save_dGK = [&](bool Is_even=true) {
//         if constexpr (!IsGating) {
//             return;
//         }
//         Tensor cdGK = make_identity_tensor(make_shape(size<0>(sdGK)));
//         Tensor tdGKcdGK = gmem_thr_copy_dLogG.partition_D(cdGK);
//         Tensor tdGKsdGK = gmem_thr_copy_dLogG.partition_S(sdGK);
//         Tensor tdGKgdGK = gmem_thr_copy_dLogG.partition_D(gdGK);
//         if constexpr (kNThreads <= kBlockN) {
//             power::copy1d<Is_even_MN, false, kBlockN>(gmem_tiled_copy_dLogG, tdGKsdGK, tdGKgdGK, tdGKcdGK, binfo.actual_seqlen_k - n_block * kBlockN);
//         } else if (tidx < kBlockN) {
//             power::copy1d<Is_even_MN, false, kBlockN>(gmem_tiled_copy_dLogG, tdGKsdGK, tdGKgdGK, tdGKcdGK, binfo.actual_seqlen_k - n_block * kBlockN);
//         }
//     };
//     auto load_dy = [&](const int m_block, const bool Is_even=true, const bool Clear_OOB_MN=true) {
//         if (!Is_even) { // if we are handling the last block of Q
//             Tensor cdy = make_identity_tensor(make_shape(size<0>(sdy)));    // (BLK_M) -> (blk_m)
//             Tensor tdycdy = gmem_thr_copy_dy.partition_D(cdy);
//             if constexpr (kNThreads <= kBlockM) {
//                 if (Clear_OOB_MN) {
//                     power::copy1d<Is_even_MN, true, kBlockM>(gmem_tiled_copy_dy, tdygdy, tdysdy, tdycdy, binfo.actual_seqlen_q - m_block * kBlockM);
//                 } else {
//                     power::copy1d<Is_even_MN, false, kBlockM>(gmem_tiled_copy_dy, tdygdy, tdysdy, tdycdy, binfo.actual_seqlen_q - m_block * kBlockM);
//                 }
//             } else if (tidx < kBlockM) {
//                 if (Clear_OOB_MN) {
//                     power::copy1d<Is_even_MN, true, kBlockM>(gmem_tiled_copy_dy, tdygdy, tdysdy, tdycdy, binfo.actual_seqlen_q - m_block * kBlockM);
//                 } else {
//                     power::copy1d<Is_even_MN, false, kBlockM>(gmem_tiled_copy_dy, tdygdy, tdysdy, tdycdy, binfo.actual_seqlen_q - m_block * kBlockM);
//                 }
//             }
//         } else { // nice even copy
//             if constexpr (kNThreads <= kBlockM) {
//                 cute::copy(gmem_tiled_copy_dy, tdygdy, tdysdy);
//             } else if (tidx < kBlockM) {
//                 cute::copy(gmem_tiled_copy_dy, tdygdy, tdysdy);
//             }
//         }
//     };
//     auto bump_dy = [&]() {
//         tdygdy.data() = tdygdy.data() + index_t(-kBlockM * params.dy_row_stride);
//     };
//     auto load_dY = [&](const int m_block, const bool Is_even=true, const bool Clear_OOB_MN=true) {
//         if (!Is_even || !Is_even_K) { // if we are handling the last block of dY or k is not divisible by 16
//             if (Clear_OOB_MN) {
//                 power::copy<Is_even_MN, Is_even_K, true>(gmem_tiled_copy_QKV, tdYgdY, tdYsdY, tQdYcQdY, tQdYpQdY, binfo.actual_seqlen_q - m_block * kBlockM);
//             } else {
//                 power::copy<Is_even_MN, Is_even_K, false>(gmem_tiled_copy_QKV, tdYgdY, tdYsdY, tQdYcQdY, tQdYpQdY, binfo.actual_seqlen_q - m_block * kBlockM);
//             }
//         } else { // nice even copy
//             cute::copy(gmem_tiled_copy_QKV, tdYgdY, tdYsdY);
//         }
//     };
//     auto bump_dY = [&]() {
//         tdYgdY.data() = tdYgdY.data() + index_t(-kBlockM * params.dY_row_stride);
//     };
//     auto load_Q = [&](const int m_block, const bool Is_even_round=true, const bool Clear_OOB_MN=true) {
//         if (!Is_even_round || !Is_even_K) { // if we are handling the last block of Q or K is not divisible by 16
//             if (Clear_OOB_MN) {
//                 power::copy<Is_even_MN, Is_even_K, true>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQdYcQdY, tQdYpQdY, binfo.actual_seqlen_q - m_block * kBlockM);
//             } else {
//                 power::copy<Is_even_MN, Is_even_K, false>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQdYcQdY, tQdYpQdY, binfo.actual_seqlen_q - m_block * kBlockM);
//             }
//         } else { // nice even copy
//             cute::copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);
//         }
//     };
//     auto bump_Q = [&]() {
//         tQgQ.data() = tQgQ.data() + index_t(-kBlockM * params.q_row_stride);
//     };
//     auto load_K = [&](bool Is_even=true, bool Clear_OOB_MN=true) {
//         if (!Is_even || !Is_even_K) { // if we are handling the last block of Q or K is not divisible by 16
//             if (Clear_OOB_MN) {
//                 power::copy<Is_even_MN, Is_even_K, true>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN);
//             } else {
//                 power::copy<Is_even_MN, Is_even_K, false>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN);
//             }
//         } else { // nice even copy
//             cute::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
//         }
//     };
//     auto load_V = [&](bool Is_even=false, bool Clear_OOB_MN=true) {
//         if (!Is_even || !Is_even_K) { // if we are handling the last block of Q or K is not divisible by 16
//             if (Clear_OOB_MN) {
//                 power::copy<Is_even_MN, Is_even_K, true>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN);
//             } else {
//                 power::copy<Is_even_MN, Is_even_K, false>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN);
//             }
//         } else { // nice even copy
//             cute::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
//         }
//     };

//     // debugger
//     #define DEBUG_PRINT(acc, name) \
//         if (true) { \
//             auto r_acc = make_tensor(acc.data(), acc.layout()); \
//             Tensor t_acc = smem_thr_copy_PdS.retile_S(r_acc); \
//             cute::copy(smem_tiled_copy_PdS, t_acc, tdSsdS); \
//             __syncthreads(); \
//             if (DEBUGGER_THREAD) { \
//                 printf("dKdV KERNEL: m_block = %d, %s:\n", m_block, name); \
//                 print_tensor(sdS); \
//                 printf("\n"); \
//             } \
//             __syncthreads(); \
//         }

//     // Prologue
//     // Register for masking
//     Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});    // (BLK_M,BLK_N) -> (blk_m,blk_n)
//     Tensor taccScS = thr_mma_sdp.partition_C(caccS);                           // (MMA,MMA_N,MMA_N)
//     static_assert(decltype(size<0>(taccScS))::value == 4);
//     // Convert to ((2, 2), MMA_M, MMA_N) then take only the row indices.
//     Tensor taccScS_row = logical_divide(taccScS, Shape<_2>{})(make_coord(0, _), _, 0);

//     // Initialize double buffer position for sQ
//     if (Double_buffer && m_block % 2 == 1) { 
//         tQsQ.data() = tQsQ.data() + size(sQ);
//         tSsQ.data() = tSsQ.data() + size(sQ);
//         tdKsQt.data() = tdKsQt.data() + size(sQ);
//     }

//     if (Kernel_traits::Is_V_in_regs) {
//         // if Is_V_in_regs, start with loading V
//         load_V(!is_last_n_block);
//         cute::cp_async_fence();
//     }

//     // load Q, K
//     load_Q(m_block, false); // start from last Q block, so OOB possible
//     load_K(!is_last_n_block);
//     if (!Kernel_traits::Is_V_in_regs) {
//         load_V(!is_last_n_block);
//     }
//     cute::cp_async_fence();

//     // load G_Q, G_K
//     load_GQ(m_block, false);
//     load_GK(!is_last_n_block);
//     cute::cp_async_fence();

//     // load dY
//     load_dY(m_block, false);
//     cute::cp_async_fence();

//     // load dy
//     load_dy(m_block, false);
//     cute::cp_async_fence();

//     // dummy fence
//     cute::cp_async_fence();

//     if (Kernel_traits::Is_V_in_regs) { // load V into registers so that we can use sV for sS
//         cute::cp_async_wait<4>();
//         __syncthreads();
//         Tensor tdPrV_copy_view = smem_thr_copy_KV.retile_D(tdPrV);
//         CUTE_STATIC_ASSERT_V(size<1>(tdPsV) == size<1>(tdPrV_copy_view));            // M
//         cute::copy(smem_tiled_copy_KV, tdPsV, tdPrV_copy_view);
//     }

//     // clear accumulators
//     clear(acc_dk);
//     clear(acc_dv);

//     // Registers for storing dGK
//     using acc_tensor = decltype(partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{}));
//     using rowcol_layout = decltype(power::convert_layout_acc_rowcol(acc_tensor{}.layout()));
//     Tensor acc_dGK = make_tensor<float>(Shape<Int<(IsGating ? size<1>(rowcol_layout{}) : 0)>>{});
//     clear(acc_dGK);

//     for (; m_block >= m_block_min; --m_block) {
//         Tensor acc_s = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
//         clear(acc_s);

//         // start loading Q[i+1] from the get-go
//         if (Double_buffer && m_block > m_block_min) {
//             tQsQ.data() = tQsQ.data() + ((m_block - 1) % 2 == 0 ? -size(sQ) : size(sQ));
//             bump_Q();
//             load_Q(m_block - 1, false);
//         }
//         cute::cp_async_fence();

//         // makes sure Q, K, V are loaded
//         cute::cp_async_wait<5>();
//         __syncthreads();

//         // compute S = QK^T
//         power::gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma_sdp,
//                     smem_tiled_copy_QdY, smem_tiled_copy_KV, smem_thr_copy_QdY, smem_thr_copy_KV);


//         // bump tSsQ, we are done with it
//         if (Double_buffer && m_block > m_block_min) {
//             tSsQ.data() = tSsQ.data() + ((m_block - 1) % 2 == 0 ? -size(sQ) : size(sQ));
//         }

//         // log S
// #ifdef DEBUG_POWER_BWD_DKDV
//         DEBUG_PRINT(acc_s, "S = QK^T")
// #endif
        
//         // Reshape acc_s from (MMA=4, MMA_N, MMA_N) to (row=(2, MMA_N), col=(2, MMA_N))
//         Tensor scores = make_tensor(acc_s.data(), power::convert_layout_acc_rowcol(acc_s.layout()));

//         // Keep S
//         Tensor S_rowcol = make_tensor_like(scores);
//         #pragma unroll
//         for (int i = 0; i < size(scores); ++i) {
//             S_rowcol(i) = scores(i);
//         }

//         // Apply abslogp, get T[i] = p * log(abs(S[i]) + ε)
//         power::apply_abslogp<is_bf16>(scores, params.ε);

// #ifdef DEBUG_POWER_BWD_DKDV
//         DEBUG_PRINT(acc_s, "after abslogp")
// #endif

//         // wait for G_Q[i] (G_K as well for the first iteration)
//         cute::cp_async_wait<4>();
//         __syncthreads();


//         // Apply gating, get Z[i] = T[i] + p * (G_Q[i] @ 1^T - 1 @ G_K^T)
//         if constexpr (IsGating) {
//             power::apply_gating</*masked=*/false, typename Kernel_traits::TiledMmaSdP, MMA_N_SdP>(acc_s, sGQ, sGK, params.deg);
//         }

// #ifdef DEBUG_POWER_BWD_DKDV
//         DEBUG_PRINT(acc_s, "after gating")
// #endif

//         // Start loading G_Q[i+1]
//         if (IsGating && m_block > m_block_min) {
//             __syncthreads();
//             bump_GQ();
//             load_GQ(m_block - 1, false);
//         }
//         cute::cp_async_fence();

//         // Keep Z[i]
//         Tensor Z_rowcol = make_tensor_like(scores);
//         #pragma unroll
//         for (int i = 0; i < size(scores); ++i) {
//             Z_rowcol(i) = scores(i);
//         }

//         // Apply mask, get Z[i] * M
//         if (!Is_causal) {
//             if (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k) {
//                 power::apply_mask(scores, binfo.actual_seqlen_k,
//                                   n_block * kBlockN + (tidx / 32 / AtomLayoutMS) * MMA_N_SdP * 16);
//             }
//         } else {
//             if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k
//                 || (!Is_even_MN && ((n_block + 1) * kBlockN >= binfo.actual_seqlen_k) || (m_block + 1) * kBlockM >= binfo.actual_seqlen_q)) {
//                 power::apply_mask_causal(scores, n_block * kBlockN + (tidx / 32 / AtomLayoutMS) * MMA_N_SdP * 16,
//                                          binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
//                                          binfo.actual_seqlen_q,
//                                          // binfo.actual_seqlen_k, m_block * kBlockM + (tidx / 32) % AtomLayoutMS * 16 + (tidx % 32) / 4,
//                                          AtomLayoutMS * 16);
//             }
//         }

// #ifdef DEBUG_POWER_BWD_DKDV
//         DEBUG_PRINT(acc_s, "after mask")
// #endif

//         // Apply exp, get P[i] = exp(Z * M)
//         power::scale_apply_exp2(scores, params.scale_softmax_log2);

// #ifdef DEBUG_POWER_BWD_DKDV
//         DEBUG_PRINT(acc_s, "after exp")
// #endif

//         // Convert P[i] to fp16/bf16
//         Tensor rP = power::convert_type<Element>(acc_s);
//         // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_N, MMA_N / 2)
//         // if using m16n8k16 or (4, MMA_N, MMA_N) if using m16n8k8.
//         Tensor tPrP = make_tensor(rP.data(), power::convert_layout_acc_Aregs<Kernel_traits::TiledMmaSdP>(rP.layout()));
//         Tensor tPaP = smem_thr_copy_PdS.retile_S(tPrP);     // ((Atom,AtomNum), MMA_N, MMA_N)
//         // put P[i] back to smem
//         cute::copy(smem_tiled_copy_PdS, tPaP, tPsP);

//         // Wait for dY[i]
//         cute::cp_async_wait<4>();
//         __syncthreads();

// #ifdef DEBUG_POWER_BWD_DKDV
//         if (DEBUGGER_THREAD) {
//             printf("m_block = %d, acc_dv before gemm:\n", m_block);
//             for (int i = 0; i < size(acc_dv); ++i) {
//                 printf("%f ", acc_dv[i]);
//             }
//             printf("\n");
//         }
// #endif

//         // compute dV[i] = P^T @ dY
//         power::gemm(acc_dv, tdVrPt, tdVrdY, tdVsPt, tdVsdYt, tiled_mma_dkv,
//                     smem_tiled_copy_PdSt, smem_tiled_copy_QdYt, smem_thr_copy_PdSt, smem_thr_copy_QdYt);

// #ifdef DEBUG_POWER_BWD_DKDV
//         if (DEBUGGER_THREAD) {
//             printf("m_block = %d, acc_dv after gemm:\n", m_block);
//             for (int i = 0; i < size(acc_dv); ++i) {
//                 printf("%f ", acc_dv[i]);
//             }
//             printf("\n");
//         }
// #endif

//         // Starting computing dP[i] = dY[i] @ V + dy[i] @ 1^T
//         Tensor acc_dp = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
//         CUTE_STATIC_ASSERT_V(size<0>(acc_dp) == size<0>(acc_s));                     // MMA
//         CUTE_STATIC_ASSERT_V(size<1>(acc_dp) == size<1>(acc_s));                     // MMA
//         CUTE_STATIC_ASSERT_V(size<2>(acc_dp) == size<2>(acc_s));                     // MMA
//         clear(acc_dp);

// #ifdef DEBUG_POWER_BWD_DKDV
//         if (DEBUGGER_THREAD) {
//             printf("m_block = %d, sdY: \n", m_block);
//             print_tensor(sdY);
//             printf("\n");
//             printf("m_block = %d, sV: \n", m_block);
//             print_tensor(sV);
//             printf("\n");
//         }
// #endif

//         // Compute dP[i] = dY[i] @ V
//         power::gemm</*A_in_regs=*/false, /*B_in_regs=*/Kernel_traits::Is_V_in_regs>(
//             acc_dp, tdPrdY, tdPrV, tdPsdY, tdPsV, tiled_mma_sdp,
//             smem_tiled_copy_QdY, smem_tiled_copy_KV, smem_thr_copy_QdY, smem_thr_copy_KV
//         );

//         // start loading dY[i+1]
//         if (m_block > m_block_min) {
//             __syncthreads();
//             bump_dY();
//             load_dY(m_block - 1, false);
//         }
//         cute::cp_async_fence();

//         // wait for dy[i]
//         cute::cp_async_wait<4>();
//         __syncthreads();

// #ifdef DEBUG_POWER_BWD_DKDV
//         DEBUG_PRINT(acc_dp, "dP = dY @ V")
// #endif

//         // read in dy from smem
//         Tensor rdy = make_tensor<float>(Shape<Int<size<1>(acc_dp) * 2>>{});  // (MMA_M * 2)
//         for (int mi = 0; mi < size(rdy); ++mi) {
//             rdy(mi) = sdy(power::mi_to_m<Kernel_traits::TiledMmaSdP>(mi));
//         }

//         // Reshape acc_dp from (MMA=4, MMA_N, MMA_N) to (row=(2, MMA_N), col=(2, MMA_N))
//         Tensor dZ = make_tensor(acc_dp.data(), scores.layout());

//         // Compute dZ, or dT
//         CUTE_UNROLL
//         for (int mi = 0; mi < size<0>(dZ); ++mi) {
//             CUTE_UNROLL
//             for (int ni = 0; ni < size<1>(dZ); ++ni) {
//                 dZ(mi, ni) += rdy(mi); // add dy, dP = dY @ V + dy @ 1^T
//                 dZ(mi, ni) = dZ(mi, ni) * scores(mi, ni) * params.deg; // dZ = dP * P * p
//             }
//         }

// #ifdef DEBUG_POWER_BWD_DKDV
//         DEBUG_PRINT(acc_dp, "dZ * p");
// #endif

//         // negative sum dZ across cols to get dGK
//         if constexpr (IsGating) {
//             CUTE_UNROLL
//             for (int ni = 0; ni < size<1>(dZ); ++ni) {
//                 CUTE_UNROLL
//                 for (int mi = 0; mi < size<0>(dZ); ++mi) {
//                     acc_dGK(ni) -= dZ(mi, ni);
//                 }
//             }
//         }

// #ifdef DEBUG_POWER_BWD_DKDV
//         if (DEBUGGER_THREAD) {
//             printf("m_block = %d, acc_dGK:\n", m_block);
//             for (int i = 0; i < size(acc_dGK); ++i) {
//                 printf("%f ", acc_dGK[i]);
//             }
//             printf("\n");
//         }
//         __syncthreads();
// #endif

//         // start loading dy[i+1]
//         if (m_block > m_block_min) {
//             __syncthreads();
//             bump_dy();
//             load_dy(m_block - 1);
//         }
//         cute::cp_async_fence();

//         // Compute dS = dT * p * sign(S) / (abs(S) + ε)
//         Tensor dS = make_tensor(dZ.data(), dZ.layout());
//         CUTE_UNROLL
//         for (int i = 0; i < size(dZ); ++i) {
//             dS(i) = (S_rowcol(i) < 0 ? -dZ(i) : dZ(i)) / (cuda_abs(S_rowcol(i)) + params.ε); 
//         }

//         // copy dS to smem
//         Tensor dS_reshaped = make_tensor(dS.data(), acc_dp.layout());
//         // Convert dS from fp32 to fp16
//         Tensor tdSrdS = power::convert_type<Element>(dS_reshaped);
//         // if (cute::thread0()) { print(tPrP); }
//         Tensor tdSadS = smem_thr_copy_PdS.retile_S(tdSrdS);       // ((Atom,AtomNum), MMA_N, MMA_N)
//         cute::copy(smem_tiled_copy_PdS, tdSadS, tdSsdS);
//         __syncthreads();

// #ifdef DEBUG_POWER_BWD_DKDV
//         DEBUG_PRINT(acc_dp, "dS")
// #endif


//         // Compute dK[i] = dS^T @ Q
//         power::gemm(acc_dk, tdKrdSt, tdKrQt, tdKsdSt, tdKsQt, tiled_mma_dkv,
//                     smem_tiled_copy_PdSt, smem_tiled_copy_QdYt, smem_thr_copy_PdSt, smem_thr_copy_QdYt);

//         // Bump tdKsQt, we are done with it
//         if (Double_buffer && m_block > m_block_min) {
//             tdKsQt.data() = tdKsQt.data() + (m_block % 2 == 0 ? size(sQ) : -size(sQ));
//         }

//         // if not double buffer, this is where we start loading next Q
//         if (!Double_buffer && m_block > m_block_min) {
//             __syncthreads();
//             // Advance gQ
//             bump_Q();
//             load_Q(m_block - 1, false);
//         }
//         power::cp_async_fence();
//     }

//     // Epilogue

//     // Sum acc_dGK
//     if constexpr (IsGating) {
//         SumOp<float> sum_op;
//         power_attention::quad_allreduce_mod_(acc_dGK, acc_dGK, sum_op); // sum over threads that have the same mod 4
//         const int warp_id = tidx / 32;
//         constexpr int warp_group_max = Kernel_traits::AtomLayoutMSdP - 1;
//         // warp reduce
//         for (int warp_group = warp_group_max; warp_group >= 0; --warp_group) {
//             if (warp_id % Kernel_traits::AtomLayoutMSdP == warp_group) {
//                 if ((tidx % 32) / 4 == 0) {
//                     for (int ni = 0; ni < size(acc_dGK); ++ni) {
//                         int n = ni_to_n_warpNcontiguous<Kernel_traits::TiledMmaSdP, MMA_N_SdP>(ni);
//                         sdGK[n] = warp_group == warp_group_max ? acc_dGK(ni) : acc_dGK(ni) + sdGK[n];
//                     }
//                 }
//             }
//             __syncthreads();
//         }
//         // write dGK back to global memory
//         save_dGK();
//     }

// #ifdef DEBUG_POWER_BWD_DKDV
//     if (DEBUGGER_THREAD) {
//         printf("acc_dk:\n");
//         print_tensor(acc_dk);
//         printf("\n");
//         printf("acc_dv:\n");
//         print_tensor(acc_dv);
//         printf("\n");
//     }
// #endif
//     // Convert acc_dv from fp32 to fp16
//     Tensor rdK = make_tensor<Element>(acc_dk.layout());
//     Tensor rdV = make_tensor<Element>(acc_dv.layout());
//     power::convert_type(acc_dk, rdK);
//     power::convert_type(acc_dv, rdV);

// #ifdef DEBUG_POWER_BWD_DKDV
//     if (DEBUGGER_THREAD) {
//         printf("rdK:\n");
//         print_tensor(rdK);
//         printf("\n");
//         printf("rdV:\n");
//         print_tensor(rdV);
//         printf("\n");
//     }
// #endif

//     Tensor sdK = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutdKV{});  // (SMEM_N, SMEM_K)
//     Tensor sdV = make_tensor(sdK.data() + size(sdK), typename Kernel_traits::SmemLayoutdKV{}); // (SMEM_N, SMEM_K)

//     // Partition sdV and sdK to match the accumulator partitioning
//     auto smem_tiled_copy_dKV = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdKV{}, tiled_mma_dkv);
//     auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(tidx);
//     Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(rdK);       // ((Atom,AtomNum), MMA_N, MMA_N)
//     Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(sdK);   // ((Atom,AtomNum),PIPE_M,PIPE_N)
//     Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(rdV);       // ((Atom,AtomNum), MMA_N, MMA_N)
//     Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(sdV);    // ((Atom,AtomNum),PIPE_M,PIPE_N)

//     // We need syncthreads here since we're writing to the same location as sK and sV.
//     // Without syncthreads, some thread might modify the location of sK while another thread
//     // is reading it for dQ gemm, leading to a race condition.
//     // If Is_last, there's already a __syncthreads() at the end of the loop.
//     __syncthreads();

// #ifdef DEBUG_POWER_BWD_DKDV
//     if (DEBUGGER_THREAD) {
//         printf("taccdKrdK:\n");
//         print_tensor(taccdKrdK);
//         printf("\n");
//         printf("taccdVrdV:\n");
//         print_tensor(taccdVrdV);
//         printf("\n");
//     }
// #endif

//     // copy dK and dV back to smem
//     cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);
//     cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);

//     const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
//        + n_block * kBlockN * params.dk_row_stride + (bidh / params.h_h_k_ratio) * params.dk_head_stride;
//     const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
//        + n_block * kBlockN * params.dv_row_stride + (bidh / params.h_h_k_ratio) * params.dv_head_stride;
//     Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
//                              Shape<Int<kBlockN>, Int<kHeadDim>>{},
//                              make_stride(params.dk_row_stride, _1{}));
//     Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
//                              Shape<Int<kBlockN>, Int<kHeadDim>>{},
//                              make_stride(params.dv_row_stride, _1{}));

//     typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
//     auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
//     Tensor tdKsdK = gmem_thr_copy_dKV.partition_S(sdK);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
//     Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
//     Tensor tdVsdV = gmem_thr_copy_dKV.partition_S(sdV);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
//     Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);

//     __syncthreads();

// #ifdef DEBUG_POWER_BWD_DKDV
//     if (DEBUGGER_THREAD) {
//         printf("sdK:\n");
//         print_tensor(sdK);
//         printf("\n");
//         printf("sdV:\n");
//         print_tensor(sdV);
//         printf("\n");
//     }
// #endif

//     // copy dK and dV back to gmem
//     Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
//     cute::copy(gmem_tiled_copy_dKV, tdKsdK, tdKrdK);
//     Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
//     cute::copy(gmem_tiled_copy_dKV, tdVsdV, tdVrdV);
//     Tensor cdKV = make_identity_tensor(make_shape(size<0>(sdK), size<1>(sdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
//     Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
//     Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
//     #pragma unroll
//     for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d; }
//     // Clear_OOB_K must be false since we don't want to write zeros to gmem
//     power::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
//         gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
//     );
//     power::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
//         gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
//     );

// }


// ////////////////////////////////////////////////////////////////////////////////////////////////////

// template<typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool IsGating, typename Params>
// inline __device__ void compute_dq_1rowblock(const Params &params, const int bidb, const int bidh, const int m_block) {
//     // Math:
//     // Forward:
//     // S = QK^T
//     // M = Causal Mask
//     // T = p * log(abs(S) + ε)
//     // Z = T + p * (G_Q @ 1^T - 1 @ G_K^T)
//     // P = exp(Z * M)
//     // Y = P @ V
//     // y = P @ 1

//     // Backward:
//     // dV = P^T @ dY
//     // dP = dY @ V + dy @ 1^T
//     // dZ = dP * P
//     // dT = dZ
//     // dS = dT * p * sign(S) / (abs(S) + ε)
//     // dK = dS^T @ Q
//     // dQ = dS @ K
//     // dG_Q = (dZ * p) @ 1
//     // dG_K = - (dZ * p)^T @ 1

//     // Main steps (dQ)
//     // Prologue:
//     //     * load Q, load K[0]
//     //     * load G_Q, G_K[0]
//     //     * load dY, dy, load V[0]
//     //     * dummy load
//     // Main loop: for each iteration i
//     //     * Start loading K[i+1]
//     //     * Wait<4> for K[i] (Q)
//     //     * Compute S = QK^T
//     //     * Compute T = p * log(abs(S) + ε)
//     //     * Wait<3> for G_K[i] (G_Q)
//     //     * Compute Z = T + p * (G_Q @ 1^T - 1 @ G_K^T)
//     //     * Start loading G_K[i+1]
//     //     * Compute P = exp(Z * M)
//     //     * Wait<3> for V[i] (dY, dy)
//     //     * Compute dP = dY @ V
//     //     * Start loading V[i+1]
//     //     * Compute dP = dP + dy @ 1^T
//     //     * Compute dZ = dP * P
//     //     * Compute dG_Q = (dZ * p) @ 1
//     //     * Compute dS = dT * p * sign(S) / (abs(S) + ε)
//     //     * Compute dQ = dS @ K
//     //     * [if No_double_buffer] load K[i+1]
//     // Epilogue:
//     //     * reduce acc_dGQ inter warp and intra warp, put in smem, write to gmem
//     //     * put acc_dQ in smem, write to gmem


//     using Element = typename Kernel_traits::Element;
//     using ElementAccum = typename Kernel_traits::ElementAccum;
//     using index_t = typename Kernel_traits::index_t;

//     // Shared memory.
//     extern __shared__ char smem_[];

//     // The thread index.
//     const int tidx = threadIdx.x;

//     constexpr int kBlockM = Kernel_traits::kBlockM;
//     constexpr int kBlockN = Kernel_traits::kBlockN;
//     constexpr int kHeadDim = Kernel_traits::kHeadDim;
//     constexpr int MMA_N_SdP = kBlockN / decltype(typename Kernel_traits::TiledMmaSdP{}.template tile_size_mnk<1>())::value; /*Number of tiles along the N dimension*/
//     constexpr int kNWarps = Kernel_traits::kNWarps;
//     constexpr int AtomLayoutMS = Kernel_traits::AtomLayoutMSdP;
//     constexpr int AtomLayoutNS = kNWarps / AtomLayoutMS;
//     constexpr bool Double_buffer = !Kernel_traits::No_double_buffer;
//     constexpr int kNThreads = Kernel_traits::kNThreads;
//     constexpr bool is_bf16 = std::is_same_v<Element, cutlass::bfloat16_t>;

//     const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
//     if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

//     bool is_last_m_block = m_block == (cute::ceil_div(binfo.actual_seqlen_q, kBlockM) - 1);

//     const index_t offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
//         + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
//     const index_t offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
//         + (bidh / params.h_h_k_ratio) * params.k_head_stride;
//     const index_t offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
//         + (bidh / params.h_h_k_ratio) * params.v_head_stride;
//     const index_t offset_gq = binfo.q_offset(params.g_batch_stride, params.g_row_stride, bidb)
//         + m_block * kBlockM * params.g_row_stride + static_cast<index_t>(bidh);
//     const index_t offset_gk = binfo.k_offset(params.g_batch_stride, params.g_row_stride, bidb)
//         + static_cast<index_t>(bidh); // we want bidh instead of bidh / params.h_h_k_ratio for special handling for GQA
//     const index_t offset_dY = binfo.q_offset(params.dY_batch_stride, params.dY_row_stride, bidb)
//         + m_block * kBlockM * params.dY_row_stride + bidh * params.dY_head_stride;
//     const index_t offset_dy = binfo.q_offset(params.dy_batch_stride, params.dy_row_stride, bidb)
//         + m_block * kBlockM * params.dy_row_stride + bidh * params.dy_head_stride;

//     ////////////////Global Memory////////////////////
//     Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + offset_q),
//                             Shape<Int<kBlockM>, Int<kHeadDim>>{},
//                             make_stride(params.q_row_stride, _1{}));
//     Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + offset_k),
//                             Shape<Int<kBlockN>, Int<kHeadDim>>{},
//                             make_stride(params.k_row_stride, _1{}));
//     Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + offset_v),
//                             Shape<Int<kBlockN>, Int<kHeadDim>>{},
//                             make_stride(params.v_row_stride, _1{}));
//     Tensor gGQ = make_tensor(make_gmem_ptr(reinterpret_cast<float *>(IsGating ? params.log_g_q_ptr : params.q_ptr) + offset_gq), 
//                             Shape<Int<kBlockM>>{},
//                             make_stride(params.g_row_stride));
//     // Use the same gating factor for Q and K here to increase L2 cache hits
//     // Only valid for training
//     Tensor gGK = make_tensor(make_gmem_ptr(reinterpret_cast<float *>(IsGating ? params.log_g_k_ptr : params.k_ptr) + offset_gk),
//                              Shape<Int<kBlockN>>{},
//                              make_stride(params.g_row_stride));
//     Tensor gdY = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dY_ptr) + offset_dY),
//                              Shape<Int<kBlockM>, Int<kHeadDim>>{},
//                              make_stride(params.dY_row_stride, _1{}));
//     Tensor gdy = make_tensor(make_gmem_ptr(reinterpret_cast<float *>(params.dy_ptr) + offset_dy),
//                             Shape<Int<kBlockM>>{},
//                             make_stride(params.dy_row_stride));
//     Tensor gdGQ = make_tensor(make_gmem_ptr(reinterpret_cast<float *>(IsGating ? params.dlog_g_q_ptr : params.q_ptr) + offset_gq),
//                              Shape<Int<kBlockM>>{},
//                              make_stride(params.g_row_stride));

//     ////////////////Shared Memory////////////////////
//     Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
//                             typename Kernel_traits::SmemLayoutQdY{});
//     Tensor sQt = make_tensor(sQ.data(),
//                             typename Kernel_traits::SmemLayoutQdYtransposed{});
//     Tensor sQtNoSwizzle = make_tensor(sQ.data(),
//                             typename Kernel_traits::SmemLayoutQdYtransposedNoSwizzle{});
//     // Double buffer for sQ
//     Tensor sdY = make_tensor(sQ.data() + size(sQ),
//                              typename Kernel_traits::SmemLayoutQdY{});
//     Tensor sdYt = make_tensor(sdY.data(),
//                              typename Kernel_traits::SmemLayoutQdYtransposed{});
//     Tensor sdYtransposedNoSwizzle = make_tensor(sdY.data(),
//                                                 typename Kernel_traits::SmemLayoutQdYtransposedNoSwizzle{});
//     Tensor sdy = make_tensor(make_smem_ptr(reinterpret_cast<float *>(&(*sdY.data()) +size(sdY))),
//                              typename Kernel_traits::SmemLayoutdy{});
//     Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(&(*sdy.data()) + size(sdy))),
//                             typename Kernel_traits::SmemLayoutKV{});
//     Tensor sKt = make_tensor(sK.data(),
//                             typename Kernel_traits::SmemLayoutKtransposed{});
//     Tensor sKtNoSwizzle = make_tensor(sK.data(),
//                             typename Kernel_traits::SmemLayoutKtransposedNoSwizzle{});
//     Tensor sV = make_tensor(sK.data() + (Double_buffer ? 2 : 1) * size(sK),
//                             typename Kernel_traits::SmemLayoutKV{});
//     Tensor sdS = make_tensor(sV.data() + size(sV), // impossible to let V stay in registers
//                              typename Kernel_traits::SmemLayoutPdS{});
//     Tensor sdSt = make_tensor(sdS.data(),
//                             typename Kernel_traits::SmemLayoutPdStransposed{});
//     Tensor sdStNoSwizzle = make_tensor(sdS.data(),
//                             typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});
//     Tensor sGQ = make_tensor(make_smem_ptr(reinterpret_cast<float *>(&(*sdS.data()) + (IsGating ? size(sdS) : 0))),
//                              typename Kernel_traits::SmemLayoutGQ{});
//     Tensor sGK = make_tensor(make_smem_ptr(reinterpret_cast<float *>(&(*sGQ.data()) + (IsGating ? size(sGQ) : 0))),
//                              typename Kernel_traits::SmemLayoutGK{});
//     Tensor sdGQ = make_tensor(sGQ.data(), typename Kernel_traits::SmemLayoutGQ{});
//     // sQ and sdQ share the same memory so be careful
//     Tensor sdQ = make_tensor(sQ.data(),
//                              typename Kernel_traits::SmemLayoutdQ{});

//     typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
//     auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
//     typename Kernel_traits::GmemTiledCopydy gmem_tiled_copy_dy;
//     auto gmem_thr_copy_dy = gmem_tiled_copy_dy.get_thread_slice(tidx);
//     typename Kernel_traits::GmemTiledCopyLogG gmem_tiled_copy_LogG;
//     auto gmem_thr_copy_LogG = gmem_tiled_copy_LogG.get_thread_slice(tidx);
//     typename Kernel_traits::GmemTiledCopydLogG gmem_tiled_copy_dLogG;
//     auto gmem_thr_copy_dLogG = gmem_tiled_copy_dLogG.get_thread_slice(tidx);

//     Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
//     Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
//     Tensor tdYgdY = gmem_thr_copy_QKV.partition_S(gdY);
//     Tensor tdYsdY = gmem_thr_copy_QKV.partition_D(sdY);
//     Tensor tdygdy = gmem_thr_copy_dy.partition_S(gdy);
//     Tensor tdysdy = gmem_thr_copy_dy.partition_D(sdy);
//     Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
//     Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
//     Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
//     Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
//     Tensor tGQgGQ = gmem_thr_copy_LogG.partition_S(gGQ);  // (GCPY, GCPY_M)
//     Tensor tGQsGQ = gmem_thr_copy_LogG.partition_D(sGQ);
//     Tensor tGKgGK = gmem_thr_copy_LogG.partition_S(gGK);  // (GCPY, GCPY_N)
//     Tensor tGKsGK = gmem_thr_copy_LogG.partition_D(sGK);

//     typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
//     auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
//     Tensor tSrQ = thr_mma_sdp.partition_fragment_A(sQ);         // (MMA,MMA_N,MMA_K)
//     Tensor tSrK = thr_mma_sdp.partition_fragment_B(sK);         // (MMA,MMA_N,MMA_K)
//     Tensor tdPrdY = thr_mma_sdp.partition_fragment_A(sdY);      // (MMA,MMA_N,MMA_K)
//     Tensor tdPrV = thr_mma_sdp.partition_fragment_B(sV);        // (MMA,MMA_N,MMA_K)

//     typename Kernel_traits::TiledMmadQ tiled_mma_dq;
//     auto thr_mma_dq = tiled_mma_dq.get_thread_slice(tidx);
//     Tensor tdQrdS = thr_mma_dq.partition_fragment_A(sdS);                      // (MMA, MMA_N, MMA_N)
//     Tensor tdQrKt = thr_mma_dq.partition_fragment_B(sKtNoSwizzle);    // (MMA, MMA_K, MMA_N)

//     Tensor acc_dq = partition_fragment_C(tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_N, MMA_K

//     //
//     // Copy Atom retiling
//     //
//     auto smem_tiled_copy_QdY = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
//     auto smem_thr_copy_QdY = smem_tiled_copy_QdY.get_thread_slice(tidx);
//     Tensor tSsQ = smem_thr_copy_QdY.partition_S(sQ);
//     Tensor tdPsdY = smem_thr_copy_QdY.partition_S(sdY);

//     // auto smem_tiled_copy_KV = make_tiled_copy_B_warpcontiguousN<MMA_N_SdP>(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
//     auto smem_tiled_copy_KV = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
//     auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);
//     Tensor tSsK = smem_thr_copy_KV.partition_S(sK);
//     Tensor tdPsV = smem_thr_copy_KV.partition_S(sV);

//     // Partition sdS to match the accumulator partitioning
//     // This has to be tiled_mma_sdp, not tiled_mma_dkv
//     // Sean: I actually don't understand why this partition is necessary and won't cause issue
//     // TODO: test it out anf fix it if it doesn't work
//     // auto smem_tiled_copy_PdS = make_tiled_copy_C_warpcontiguousN<MMA_N_SdP>(typename Kernel_traits::SmemCopyAtomPdS{}, tiled_mma_sdp);
//     auto smem_tiled_copy_PdS = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomPdS{}, tiled_mma_sdp);
//     auto smem_thr_copy_PdS = smem_tiled_copy_PdS.get_thread_slice(tidx);
//     Tensor tdSsdS = smem_thr_copy_PdS.partition_D(sdS);   // ((Atom,AtomNum),PIPE_M,PIPE_N)

//     auto smem_tiled_copy_dS = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_dq);
//     auto smem_thr_copy_dS = smem_tiled_copy_dS.get_thread_slice(tidx);
//     Tensor tdQsdS = smem_thr_copy_dS.partition_S(sdS);

//     auto smem_tiled_copy_Kt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dq);
//     auto smem_thr_copy_Kt = smem_tiled_copy_Kt.get_thread_slice(tidx);
//     Tensor tdQsKt = smem_thr_copy_Kt.partition_S(sKt);

//     auto smem_tiled_copy_dQ = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdQ{}, tiled_mma_dq);
//     auto smem_thr_copy_dQ = smem_tiled_copy_dQ.get_thread_slice(tidx);

//     int n_block = 0;
//     int n_block_max = (!Is_causal) ? cute::ceil_div(binfo.actual_seqlen_k, kBlockN) : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q) / kBlockN);

//     // Predicates for QdY
//     static_assert(size<0>(sQ) == size<0>(sdY), "Q and dY must have the same number of rows");
//     static_assert(size<1>(sQ) == size<1>(sdY), "Q and dY must have the same number of columns");
//     Tensor cQdY = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M, BLK_K) -> (blk_m, blk_k)
//     Tensor tQdYcQdY = gmem_thr_copy_QKV.partition_D(cQdY);
//     // Allocate predicate tensors for k
//     Tensor tQdYpQdY = make_tensor<bool>(make_shape(size<2>(tQsQ)));
//     Tensor cGQ = make_identity_tensor(make_shape(size<0>(sGQ)));    // (BLK_M) -> (blk_m)
//     Tensor tGQcGQ = gmem_thr_copy_LogG.partition_D(cGQ);
//     if (!Is_even_K) {
//         #pragma unroll
//         for (int k = 0; k < size(tQdYpQdY); ++k) { tQdYpQdY(k) = get<1>(tQdYcQdY(0, 0, k)) < params.d; }
//     }
//     // Predicates for KV
//     static_assert(size<0>(sK) == size<0>(sV), "K and V must have the same number of rows");
//     static_assert(size<1>(sK) == size<1>(sV), "K and V must have the same number of columns");
//     Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N, BLK_K) -> (blk_n, blk_k)
//     Tensor tKVcKV = gmem_thr_copy_QKV.partition_D(cKV);
//     // Allocate predicate tensors for k
//     Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));
//     if (!Is_even_K) {
//         #pragma unroll
//         for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
//     }

//     // Loaders and bumpers
//     auto load_GQ = [&](bool NO_OOB=true, bool Clear_OOB_MN=true) {
//         if constexpr (!IsGating) {
//             return;
//         }
//         if (Clear_OOB_MN) {
//             power::copy1d<false, true, kBlockM>(gmem_tiled_copy_LogG, tGQgGQ, tGQsGQ, tGQcGQ, std::min(binfo.actual_seqlen_q - m_block * kBlockM, kBlockM));
//         } else {
//             power::copy1d<false, false, kBlockM>(gmem_tiled_copy_LogG, tGQgGQ, tGQsGQ, tGQcGQ, std::min(binfo.actual_seqlen_q - m_block * kBlockM, kBlockM));
//         }
//     };
//     auto load_GK = [&](const int n_block, const bool NO_OOB=true, const bool Clear_OOB_MN=true) {
//         if constexpr (!IsGating) {
//             return;
//         }
//         Tensor cGK = make_identity_tensor(make_shape(size<0>(sGK)));    // (BLK_N) -> (blk_n)
//         Tensor tGKcGK = gmem_thr_copy_LogG.partition_D(cGK);
//         if (Clear_OOB_MN) {
//             power::copy1d<false, true>(gmem_tiled_copy_LogG, tGKgGK, tGKsGK, tGKcGK, std::min(binfo.actual_seqlen_k - n_block * kBlockN, kBlockN));
//         } else {
//             power::copy1d<false, false>(gmem_tiled_copy_LogG, tGKgGK, tGKsGK, tGKcGK, std::min(binfo.actual_seqlen_k - n_block * kBlockN, kBlockN));
//         }
//     };
//     auto save_dGQ = [&](bool NO_OOB=true) {
//         if constexpr (!IsGating) {
//             return;
//         }
//         Tensor cGQ = make_identity_tensor(make_shape(size<0>(sGQ)));
//         Tensor tGQcGQ = gmem_thr_copy_dLogG.partition_D(cGQ);
//         Tensor tdGQsdGQ = gmem_thr_copy_dLogG.partition_S(sdGQ);
//         Tensor tdGQgdGQ = gmem_thr_copy_dLogG.partition_D(gdGQ);
//         if constexpr (kNThreads <= kBlockM) {
//             power::copy1d<Is_even_MN, false, kBlockM>(gmem_tiled_copy_dLogG, tdGQsdGQ, tdGQgdGQ, tGQcGQ, binfo.actual_seqlen_q - m_block * kBlockM);
//         } else if (tidx < kBlockM) {
//             power::copy1d<Is_even_MN, false, kBlockM>(gmem_tiled_copy_dLogG, tdGQsdGQ, tdGQgdGQ, tGQcGQ, binfo.actual_seqlen_q - m_block * kBlockM);
//         }
//     };
//     auto bump_GK = [&]() {
//         if constexpr (!IsGating) {
//             return;
//         }
//         tGKgGK.data() = tGKgGK.data() + index_t(kBlockN * params.g_row_stride);
//     };
//     auto load_dy = [&](bool NO_OOB=true, bool Clear_OOB_MN=true) {
//         if (NO_OOB) {
//             if constexpr (kNThreads <= kBlockM) {
//                 cute::copy(gmem_tiled_copy_dy, tdygdy, tdysdy);
//             } else if (tidx < kBlockM) {
//                 cute::copy(gmem_tiled_copy_dy, tdygdy, tdysdy);
//             }
//         } else { // if we are handling the last block of Q
//             Tensor cdy = make_identity_tensor(make_shape(size<0>(sdy)));    // (BLK_M) -> (blk_m)
//             Tensor tdycdy = gmem_thr_copy_dy.partition_D(cdy);
//             if constexpr (kNThreads <= kBlockM) {
//                 if (Clear_OOB_MN) {
//                     power::copy1d<Is_even_MN, true, kBlockM>(gmem_tiled_copy_dy, tdygdy, tdysdy, tdycdy, binfo.actual_seqlen_q - m_block * kBlockM);
//                 } else {
//                     power::copy1d<Is_even_MN, false, kBlockM>(gmem_tiled_copy_dy, tdygdy, tdysdy, tdycdy, binfo.actual_seqlen_q - m_block * kBlockM);
//                 }
//             } else if (tidx < kBlockM) {
//                 if (Clear_OOB_MN) {
//                     power::copy1d<Is_even_MN, true, kBlockM>(gmem_tiled_copy_dy, tdygdy, tdysdy, tdycdy, binfo.actual_seqlen_q - m_block * kBlockM);
//                 } else {
//                     power::copy1d<Is_even_MN, false, kBlockM>(gmem_tiled_copy_dy, tdygdy, tdysdy, tdycdy, binfo.actual_seqlen_q - m_block * kBlockM);
//                 }
//             }
//         }
//     };
//     auto load_dY = [&](bool NO_OOB=true, bool Clear_OOB_MN=true) {
//         if (Is_even_K && NO_OOB) {
//             cute::copy(gmem_tiled_copy_QKV, tdYgdY, tdYsdY);
//         } else { // if we are handling the last block of dY or k is not divisible by 16
//             if (Clear_OOB_MN) {
//                 power::copy<Is_even_MN, Is_even_K, true>(gmem_tiled_copy_QKV, tdYgdY, tdYsdY, tQdYcQdY, tQdYpQdY, binfo.actual_seqlen_q - m_block * kBlockM);
//             } else {
//                 power::copy<Is_even_MN, Is_even_K, false>(gmem_tiled_copy_QKV, tdYgdY, tdYsdY, tQdYcQdY, tQdYpQdY, binfo.actual_seqlen_q - m_block * kBlockM);
//             }
//         }
//     };
//     auto load_Q = [&](bool NO_OOB=true, bool Clear_OOB_MN=true) {
//         if (Is_even_K && NO_OOB) {
//             cute::copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);
//         } else { // if we are handling the last block of Q or K is not divisible by 16
//             if (Clear_OOB_MN) {
//                 power::copy<Is_even_MN, Is_even_K, true>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQdYcQdY, tQdYpQdY, binfo.actual_seqlen_q - m_block * kBlockM);    
//             } else {
//                 power::copy<Is_even_MN, Is_even_K, false>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQdYcQdY, tQdYpQdY, binfo.actual_seqlen_q - m_block * kBlockM);    
//             }
//         }
//     };
//     auto load_K = [&](const int n_block, const bool Is_even=true, const bool Clear_OOB_MN=true) {
//         if (Is_even && Is_even_K) {
//             cute::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
//         } else { // if we are handling the last block of Q, or K is not divisible by 16
//             if (Clear_OOB_MN) {
//                 power::copy<Is_even_MN, Is_even_K, true>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN);
//             } else {
//                 power::copy<Is_even_MN, Is_even_K, false>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN);
//             }
//         }
//     };
//     auto bump_K = [&]() {
//         tKgK.data() = tKgK.data() + index_t(kBlockN * params.k_row_stride);
//     };
//     auto load_V = [&](const int n_block, const bool Is_even=false, const bool Clear_OOB_MN=true) {
//         if (Is_even && Is_even_K) {
//             cute::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
//         } else { // if we are handling the last block of Q or K is not divisible by 16
//             if (Clear_OOB_MN) {
//                 power::copy<Is_even_MN, Is_even_K, true>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN);
//             } else {
//                 power::copy<Is_even_MN, Is_even_K, false>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN);
//             }
//         }
//     };
//     auto bump_V = [&]() {
//         tVgV.data() = tVgV.data() + index_t(kBlockN * params.v_row_stride);
//     };

//     #define DEBUG_PRINT(acc, name) \
//         if (true) { \
//             auto r_acc = make_tensor(acc.data(), acc.layout()); \
//             Tensor t_acc = smem_thr_copy_PdS.retile_S(r_acc); \
//             cute::copy(smem_tiled_copy_PdS, t_acc, tdSsdS); \
//             __syncthreads(); \
//             if (DEBUGGER_THREAD) { \
//                 printf("dQ KERNEL:  m_block = %d, n_block = %d, %s:\n", m_block, n_block, name); \
//                 print_tensor(sdS); \
//                 printf("\n"); \
//             } \
//             __syncthreads(); \
//         }

//     // Prologue
//     // Register for masking
//     Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});    // (BLK_M,BLK_N) -> (blk_m,blk_n)
//     Tensor taccScS = thr_mma_sdp.partition_C(caccS);                           // (MMA,MMA_N,MMA_N)
//     static_assert(decltype(size<0>(taccScS))::value == 4);
//     // Convert to ((2, 2), MMA_N, MMA_N) then take only the row indices.
//     Tensor taccScS_row = logical_divide(taccScS, Shape<_2>{})(make_coord(0, _), _, 0);

// #ifdef DEBUG_POWER_BWD_DQ
//     if constexpr (kNThreads <= kBlockM) {
//         sdy[tidx] = gdy[tidx];
//     } else if (tidx < kBlockM) {
//         sdy[tidx] = gdy[tidx];
//     }
//     __syncthreads();
//     if (DEBUGGER_THREAD) {
//         printf("dQ KERNEL:  m_block = %d, n_block = %d, first loaded sdy\n", m_block, n_block);
//         printf("gdy address: %p\n", gdy.data());
//         printf("offset_dy: %d\n", int(offset_dy));
//         printf("gdy[0]: %f\n", gdy[0]);
//         print_tensor(sdy);
//         printf("\n");
//     }
//     __syncthreads();
// #endif

//     // load Q, K
//     load_Q(!is_last_m_block);
//     load_K(n_block, false); // start from first K block, always even
//     cute::cp_async_fence();
    
//     // load G_Q, G_K
//     load_GQ(!is_last_m_block);
//     load_GK(n_block, false);
//     cute::cp_async_fence();

//     // load V
//     load_V(n_block, false);

//     // load dY
//     load_dY(!is_last_m_block);

//     // load dy
//     load_dy(!is_last_m_block);
//     cute::cp_async_fence();

//     // dummy load for No_Double_buffer cases
//     cute::cp_async_fence();

//     // clear accumulators
//     clear(acc_dq);

//     // Registers for storing dGK
//     using acc_tensor = decltype(partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{}));
//     using rowcol_layout = decltype(power::convert_layout_acc_rowcol(acc_tensor{}.layout()));
//     Tensor acc_dGQ = make_tensor<float>(Shape<Int<(IsGating ? size<0>(rowcol_layout{}) : 0)>>{});
//     clear(acc_dGQ);

//     for (; n_block <= n_block_max; ++n_block) {
//         Tensor acc_s = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
//         clear(acc_s);

//         // start loading K[i+1] from the get-go
//         if (Double_buffer && n_block < n_block_max) {
//             __syncthreads();
//             tKsK.data() = tKsK.data() + ((n_block + 1) % 2 == 0 ? -size(sK) : size(sK));
//             bump_K();
//             // The only case where uneven load is possible is when we are in the last Q block CTA 
//             // and loading the last K block
//             load_K(n_block + 1, n_block < n_block_max - 1);
//         }
//         cute::cp_async_fence();

//         // makes sure Q, K are loaded
//         cute::cp_async_wait<4>();
//         __syncthreads();

//         // compute S = QK^T
//         power::gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma_sdp,
//                     smem_tiled_copy_QdY, smem_tiled_copy_KV, smem_thr_copy_QdY, smem_thr_copy_KV);

// #ifdef DEBUG_POWER_BWD_DQ
//         DEBUG_PRINT(acc_s, "S = QK^T")
// #endif

//         // bump tSsK, we are done with it
//         if (Double_buffer && n_block < n_block_max) {
//             tSsK.data() = tSsK.data() + ((n_block + 1) % 2 == 0 ? -size(sK) : size(sK));
//         }

//         // Reshape acc_s from (MMA=4, MMA_N, MMA_N) to (row=(2, MMA_N), col=(2, MMA_N))
//         Tensor scores = make_tensor(acc_s.data(), power::convert_layout_acc_rowcol(acc_s.layout()));

//         // Keep S
//         Tensor S_rowcol = make_tensor_like(scores);
//         #pragma unroll
//         for (int i = 0; i < size(scores); ++i) {
//             S_rowcol(i) = scores(i);
//         }

//         // Apply abslogp, get T[i] = p * log(abs(S[i]) + ε)
//         if (params.use_multiplier) {
//             power::apply_abslogp<true>(scores, params.ε, params.deg, params.stabilizer);
//         } else {
//             power::apply_abslogp<false>(scores, params.ε, params.deg);
//         }

//         // wait for G_K[i] (G_Q as well for the first iteration)
//         cute::cp_async_wait<3>();
//         __syncthreads();

// #ifdef DEBUG_POWER_BWD_DQ
//         DEBUG_PRINT(acc_s, "after abslogp")
// #endif

//         // Apply gating, get Z[i] = T[i] + p * (G_Q[i] @ 1^T - 1 @ G_K^T)
//         if constexpr (IsGating) {
//             power::apply_gating</*masked=*/false, typename Kernel_traits::TiledMmaSdP>(acc_s, sGQ, sGK, params.deg);
//         }

// #ifdef DEBUG_POWER_BWD_DQ
//         DEBUG_PRINT(acc_s, "after gating")
// #endif

//         // Start loading G_K[i+1]
//         if (IsGating && n_block < n_block_max) {
//             __syncthreads();
//             bump_GK();
//             load_GK(n_block + 1, !is_last_m_block || n_block < n_block_max - 1);
//         }
//         cute::cp_async_fence();

//         // Keep Z[i]
//         Tensor Z_rowcol = make_tensor_like(scores);
//         #pragma unroll
//         for (int i = 0; i < size(scores); ++i) {
//             Z_rowcol(i) = scores(i);
//         }

//         // Apply mask, get Z[i] * M
//         // TD [2023-07-29]: I was thinking that we don't need to mask out the elements beyond
//         // actual_seqlen_k, because acc_s would be some finite value for those indices.
//         // In the end when we multiply with K to get dQ, the corresponding values of K would be 0,
//         // so the result would still be correct.
//         // However, it's possible that the values in acc_s are so large that they overflow
//         // when we multiply with dP and convert to fp16, resulting in Inf in dS and NaNs in dQ.
//         // So we need to mask out the elements beyond actual_seqlen_k.
//         if (!Is_causal) {
//             if (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k) {
//                 power::apply_mask</*LogSpace=*/true>(scores, binfo.actual_seqlen_k,
//                                   n_block * kBlockN + (tidx / 32 / AtomLayoutMS) * 8);
//             }
//         } else {
//             // Causal masking
//             if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k
//                 || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
//                 power::apply_mask_causal</*LogSpace=*/true>(scores, n_block * kBlockN + (tidx / 32 / AtomLayoutMS) * 8,
//                                          binfo.actual_seqlen_k, m_block * kBlockM + ((tidx / 32) % AtomLayoutMS) * 16 + (tidx % 32) / 4,
//                                          binfo.actual_seqlen_q,
//                                          AtomLayoutMS * 16, AtomLayoutNS * 8);
//             }
//         }

// #ifdef DEBUG_POWER_BWD_DQ
//         DEBUG_PRINT(acc_s, "after mask")
// #endif

//         // Apply exp, get P[i] = exp(Z * M)
//         power::scale_apply_exp2(scores, params.scale_softmax_log2);

// #ifdef DEBUG_POWER_BWD_DQ
//         DEBUG_PRINT(acc_s, "after exp")
// #endif

//         // Wait for V[i], dY and dy
//         cute::cp_async_wait<3>();
//         __syncthreads();

//         // Starting computing dP[i] = dY[i] @ V + dy[i] @ 1^T
//         Tensor acc_dp = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
//         CUTE_STATIC_ASSERT_V(size<0>(acc_dp) == size<0>(acc_s));                     // MMA
//         CUTE_STATIC_ASSERT_V(size<1>(acc_dp) == size<1>(acc_s));                     // MMA
//         CUTE_STATIC_ASSERT_V(size<2>(acc_dp) == size<2>(acc_s));                     // MMA
//         clear(acc_dp);

//         // Compute dP[i] = dY[i] @ V
//         power::gemm</*A_in_regs=*/false, /*B_in_regs=*/false>(
//             acc_dp, tdPrdY, tdPrV, tdPsdY, tdPsV, tiled_mma_sdp,
//             smem_tiled_copy_QdY, smem_tiled_copy_KV, smem_thr_copy_QdY, smem_thr_copy_KV
//         );

// #ifdef DEBUG_POWER_BWD_DQ
//         DEBUG_PRINT(acc_dp, "after dP = dY @ V")
// #endif

//         // Start loading V[i+1]
//         if (n_block < n_block_max) {
//             __syncthreads();
//             bump_V();
//             load_V(n_block + 1, n_block < n_block_max - 1);
//         }
//         cute::cp_async_fence();

//         // read in dy from smem
//         Tensor rdy = make_tensor<float>(Shape<Int<size<1>(acc_dp) * 2>>{});  // (MMA_M * 2)
//         for (int mi = 0; mi < size(rdy); ++mi) {
//             rdy(mi) = sdy(power::mi_to_m<Kernel_traits::TiledMmaSdP>(mi));
//         }

// #ifdef DEBUG_POWER_BWD_DQ
//         if (thread0()) {
//             print("dQ KERNEL:  m_block = %d, n_block = %d, sdy:\n", m_block, n_block);
//             print_tensor(sdy);
//             print("\n");
//         }
//         __syncthreads();
// #endif

//         // Reshape acc_dp from (MMA=4, MMA_N, MMA_N) to (row=(2, MMA_N), col=(2, MMA_N))
//         Tensor dZ = make_tensor(acc_dp.data(), scores.layout());

//         // Compute dZ, or dT
//         CUTE_UNROLL
//         for (int mi = 0; mi < size<0>(dZ); ++mi) {
//             CUTE_UNROLL
//             for (int ni = 0; ni < size<1>(dZ); ++ni) {
//                 dZ(mi, ni) += rdy(mi); // add dy, dP = dY @ V + dy @ 1^T
//                 dZ(mi, ni) = dZ(mi, ni) * scores(mi, ni); // dZ * p = dP * P * p
//                 dZ(mi, ni) += dZ(mi, ni);
//             }
//         }
//         if (params.deg == 4) {
//             CUTE_UNROLL
//             for (int i = 0; i < size(dZ); ++i) {
//                 dZ(i) += dZ(i);
//             }
//         }

// #ifdef DEBUG_POWER_BWD_DQ
//         DEBUG_PRINT(acc_dp, "dZ * p")
// #endif

//         // sum dZ * p across rows to get dGQ
//         if constexpr (IsGating) {
//             CUTE_UNROLL
//             for (int mi = 0; mi < size<0>(dZ); ++mi) {
//                 CUTE_UNROLL
//                 for (int ni = 0; ni < size<1>(dZ); ++ni) {
//                     acc_dGQ(mi) += dZ(mi, ni);
//                 }
//             }
//         }

//         // Compute dS = dT * p * sign(S) / (abs(S) + ε)
//         Tensor dS = make_tensor(dZ.data(), dZ.layout());
//         CUTE_UNROLL
//         for (int i = 0; i < size(dZ); ++i) {
//             dS(i) = (S_rowcol(i) < 0 ? -dZ(i) : dZ(i)) / (cuda_abs(S_rowcol(i)) + params.ε); 
//         }

// #ifdef DEBUG_POWER_BWD_DQ
//         DEBUG_PRINT(acc_dp, "dS")
// #endif

//         // copy dS to smem
//         Tensor dS_reshaped = make_tensor(dS.data(), acc_dp.layout());
//         // Convert dS from fp32 to fp16
//         Tensor tdSrdS = power::convert_type<Element>(dS_reshaped);
//         // if (cute::thread0()) { print(tPrP); }
//         Tensor tdSadS = smem_thr_copy_PdS.retile_S(tdSrdS);       // ((Atom,AtomNum), MMA_N, MMA_N)
//         cute::copy(smem_tiled_copy_PdS, tdSadS, tdSsdS);
//         __syncthreads();

// #ifdef DEBUG_POWER_BWD_DQ
//         if (DEBUGGER_THREAD) {
//             print("dQ KERNEL:  m_block = %d, n_block = %d, sK:\n", m_block, n_block);
//             print_tensor(sK);
//             print("\n");
//         }
//         __syncthreads();
// #endif

//         // Compute dQ[i] = dS @ K
//         power::gemm(acc_dq, tdQrdS, tdQrKt, tdQsdS, tdQsKt, tiled_mma_dq,
//                     smem_tiled_copy_dS, smem_tiled_copy_Kt, smem_thr_copy_dS, smem_thr_copy_Kt);

//         // Bump tdQsKt, we are done with it
//         if (Double_buffer && n_block < n_block_max) {
//             tdQsKt.data() = tdQsKt.data() + (n_block % 2 == 0 ? size(sK) : -size(sK));
//         }

//         // if not double buffer, this is where we start loading next K
//         if (!Double_buffer && n_block < n_block_max) {
//             __syncthreads();
//             // Advance gK
//             bump_K();
//             load_K(n_block + 1);
//         }
//         power::cp_async_fence();
//     }

//     // Epilogue

//     // Sum dGQ and dGK
//     if constexpr (IsGating) {
//         SumOp<float> sum_op;
//         quad_allreduce_(acc_dGQ, acc_dGQ, sum_op); // sum over 4 adjacent threads
//         const int warp_id = tidx / 32;
//         constexpr int warp_group_max = Kernel_traits::kNWarps / Kernel_traits::AtomLayoutMSdP - 1;
//         // cross warp reduction
//         for (int warp_group = warp_group_max; warp_group >= 0; --warp_group) {
//             if (warp_id / Kernel_traits::AtomLayoutMSdP == warp_group) {
//                 for (int mi = 0; mi < size(acc_dGQ); ++mi) {
//                     if (tidx % 4 == 0) {
//                         int m = mi_to_m<Kernel_traits::TiledMmaSdP>(mi);
//                         sdGQ[m] = warp_group == warp_group_max ? acc_dGQ(mi) : acc_dGQ(mi) + sdGQ[m];
//                     }
//                 }
//             }
//             __syncthreads();
//         }
//         // write dGQ back to global memory
//         save_dGQ();
//     }

//     // Convert acc_dq from fp32 to fp16
//     Tensor rdQ = make_tensor<Element>(acc_dq.layout());
//     power::convert_type(acc_dq, rdQ);

//     // Partition sdQ to match the accumulator partitioning
//     Tensor taccdQrdQ = smem_thr_copy_dQ.retile_S(rdQ);       // ((Atom,AtomNum), MMA_N, MMA_N)
//     Tensor taccdQsdQ = smem_thr_copy_dQ.partition_D(sdQ);   // ((Atom,AtomNum),PIPE_M,PIPE_N)

//     // We need syncthreads here since we're writing to the same location as sK and sV.
//     // Without syncthreads, some thread might modify the location of sK while another thread
//     // is reading it for dQ gemm, leading to a race condition.
//     // If Is_last, there's already a __syncthreads() at the end of the loop.
//     __syncthreads();

//     // copy dQ back to smem
//     cute::copy(smem_tiled_copy_dQ, taccdQrdQ, taccdQsdQ);

//     const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
//        + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
//     Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
//                              Shape<Int<kBlockM>, Int<kHeadDim>>{},
//                              make_stride(params.dq_row_stride, _1{}));

//     typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
//     auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
//     Tensor tdQsdQ = gmem_thr_copy_dQ.partition_S(sdQ);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
//     Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);

//     __syncthreads();
//     // copy dQ back to gmem
//     Tensor tdQrdQ = make_tensor<Element>(shape(tdQgdQ));
//     cute::copy(gmem_tiled_copy_dQ, tdQsdQ, tdQrdQ);
//     Tensor cdQ = make_identity_tensor(make_shape(size<0>(sdQ), size<1>(sdQ)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
//     Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
//     Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(tdQgdQ)));
//     #pragma unroll
//     for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
//     // Clear_OOB_K must be false since we don't want to write zeros to gmem
//     power::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
//         gmem_tiled_copy_dQ, tdQrdQ, tdQgdQ, tdQcdQ, tdQpdQ, binfo.actual_seqlen_q - m_block * kBlockM
//     );

// }

// ////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool IsGating, bool FlashEquivalent, bool NormalSpace, int Deg, typename Params>
inline __device__ void compute_dq_dk_dv_1colblock(const Params &params, const int bidb, const int bidh, const int n_block) {

    // Math:
    // Forward (power):
    // M = Causal Mask
    // S = QK^T * M
    // T = p * log(abs(S) + ε) - log_stabilizer
    // Z = T + (G_Q @ 1^T - 1 @ G_K^T)
    // P = exp(Z - Z_rowmax)
    // Y = P @ V
    // y = P @ 1

    // Foward (flash equivalent):
    // S = QK^T * M
    // T = S / stabilizer
    // Z = T + (G_Q @ 1^T - 1 @ G_K^T)
    // P = exp(Z - Z_rowmax)
    // Y = P @ V
    // y = P @ 1

    // Forward (normal space):
    // S = QK^T * M
    // T = S / stabilizer
    // Z = T * (G_Q @ 1^T - 1 @ G_K^T)^{1/p}
    // P = Z^{p}
    // Y = P @ V
    // y = P @ 1

    // Backward (power):
    // dV = P^T @ dY
    // dP = dY @ V^T + dy @ 1^T
    // dZ = dP * P
    // dT = dZ
    // dS = dT * p * sign(S) / (abs(S) + ε)
    // dK = dS^T @ Q
    // dQ = dS @ K
    // dG_Q = (dZ * p) @ 1
    // dG_K = - (dZ * p)^T @ 1

    // Options:
    // Is_V_in_regs: put V in registers instead of smem, this reduces smem pressure at the cost of regitser pressure

    // Main steps (dK dV)
    // Prologue:
    //     * [if Is_V_in_regs] start loading V
    //     * start loading K
    //     * [if not Is_V_in_regs] start loading V
    //     * start loading G_K
    //     * start loading Q[0], G_Q[0]
    //     * start loading dY[0], dy[0]
    //     * if Is_V_in_regs, wait for V, put V in registers
    //     * init acc_dV, acc_dK, acc_dGK
    // Main loop: for each iteration i
    //     * [if double buffer && i < max] start loading Q[i+1]
    //     * compute S[i] = Q[i] @ K^T, save S[i] copy
    //     * compute T[i] = p * log(abs(S[i]) + ε)
    //     * compute Z[i] = T[i] + p * (G_Q[i] @ 1^T - 1 @ G_K^T)
    //     * start loading G_Q[i+1]
    //     * compute P[i] = exp(Z[i] * M)
    //     * put P[i] in smem
    //     * compute dP[i] = dY[i] @ V
    //     * compute acc_dV += P[i]^T @ dY[i]
    //     * start loading dY[i+1]
    //     * compute dP[i] = dP[i] + dy @ 1^T
    //     * start loading dy[i+1]
    //     * compute dZ[i] = dT[i] = dP[i] * exp(Z[i] * M) # use Z[i] registers for dZ[i]
    //     * compute dS[i] = dT[i] * p * sign(S[i]) / (abs(S[i]) + ε) # use S[i] registers for dS[i]
    //     * compute acc_dGK += -(dZ[i] * p)^T @ 1
    //     * put dS[i] in smem # optionally use sV
    //     * compute acc_dK += dS[i]^T @ Q[i] # this is why Q needs double buffering, it's required both in the beginning and so late in the loop
    //     * start loading Q[i+1]
    // Epilogue:
    //     * reduce acc_dGK inter warp and intra warp, put in smem, write to gmem
    //     * put acc_dK in smem, write to gmem
    //     * put acc_dV in smem, write to gmem

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
    constexpr int MMA_N_SdP = kBlockN / decltype(typename Kernel_traits::TiledMmaSdP{}.template tile_size_mnk<1>())::value; /*Number of tiles along the N dimension*/
    constexpr int AtomLayoutMS = Kernel_traits::AtomLayoutMSdP;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int AtomLayoutNS = kNWarps / AtomLayoutMS;
    constexpr bool Double_buffer = !Kernel_traits::No_double_buffer;
    constexpr int kNThreads = Kernel_traits::kNThreads;
    constexpr int AtomLayoutNSdP = Kernel_traits::kNWarps / Kernel_traits::AtomLayoutMSdP;
    constexpr bool is_bf16 = std::is_same_v<Element, cutlass::bfloat16_t>;
    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (n_block * kBlockN >= binfo.actual_seqlen_k) return;

    int m_block = cute::ceil_div(binfo.actual_seqlen_q, kBlockM) - 1;
    bool is_last_n_block = n_block == cute::ceil_div(binfo.actual_seqlen_k, kBlockN) - 1;

    const index_t offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + n_block * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + n_block * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t offset_gq = binfo.q_offset(params.g_batch_stride, params.g_row_stride, bidb)
        + m_block * kBlockM * params.g_row_stride + bidh * params.g_head_stride;
    const index_t offset_gk = binfo.k_offset(params.g_batch_stride, params.g_row_stride, bidb)
        + n_block * kBlockN * params.g_row_stride + bidh * params.g_head_stride; // we want bidh instead of bidh / params.h_h_k_ratio for special handling for GQA
    const index_t offset_dgk = binfo.k_offset(params.dg_batch_stride, params.dg_row_stride, bidb)
        + n_block * kBlockN * params.dg_row_stride + bidh * params.dg_head_stride;
    const index_t offset_dgq = binfo.q_offset(params.dg_batch_stride, params.dg_row_stride, bidb)
        + m_block * kBlockM * params.dg_row_stride + bidh * params.dg_head_stride;
    const index_t offset_dY = binfo.q_offset(params.dY_batch_stride, params.dY_row_stride, bidb)
        + m_block * kBlockM * params.dY_row_stride + bidh * params.dY_head_stride;
    const index_t offset_dy = binfo.q_offset(params.dy_batch_stride, params.dy_row_stride, bidb)
        + m_block * kBlockM * params.dy_row_stride + bidh * params.dy_head_stride;
    const index_t offset_rowmax = binfo.q_offset(params.rowmax_batch_stride, params.rowmax_row_stride, bidb)
        + m_block * kBlockM * params.rowmax_row_stride + bidh * params.rowmax_head_stride;
    const index_t offset_dq_accum = binfo.q_offset(params.seqlen_q_rounded * params.h * params.d_rounded, params.h * params.d_rounded, bidb)
        + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride + (!params.deterministic ? 0 : blockIdx.x * params.dq_accum_split_stride);

    // if (tidx == 0 && bidb == 1 && bidh == 0) {
    //     printf("offset_dq_accum: %ld\n", offset_dq_accum);
    //     printf("params.seqlen_q_rounded: %d\n", params.seqlen_q_rounded);
    //     printf("params.h: %d\n", params.h);
    //     printf("params.d_rounded: %d\n", params.d_rounded);
    //     printf("params.dq_row_stride: %d\n", params.dq_row_stride);
    //     printf("params.dq_head_stride: %d\n", params.dq_head_stride);
    //     printf("params.dq_accum_split_stride: %d\n", params.dq_accum_split_stride);
    // }

    ////////////////Global Memory////////////////////
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));
    // gGQ won't be used if IsGating is false
    Tensor gGQ = make_tensor(make_gmem_ptr(reinterpret_cast<float *>(IsGating ? params.log_g_q_ptr : params.q_ptr) + offset_gq), 
                            Shape<Int<kBlockM>>{},
                            make_stride(params.g_row_stride));
    // We could use the same gating factor for Q and K here to increase L2 cache hits
    // but that's only valid for training
    Tensor gGK = make_tensor(make_gmem_ptr(reinterpret_cast<float *>(IsGating ? params.log_g_k_ptr : params.k_ptr) + offset_gk),
                             Shape<Int<kBlockN>>{},
                             make_stride(params.g_row_stride));
    Tensor gdY = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dY_ptr) + offset_dY),
                             Shape<Int<kBlockM>, Int<kHeadDim>>{},
                             make_stride(params.dY_row_stride, _1{}));
    Tensor gdy = make_tensor(make_gmem_ptr(reinterpret_cast<float *>(params.dy_ptr) + offset_dy),
                            Shape<Int<kBlockM>>{},
                            make_stride(params.dy_row_stride));
    Tensor growmax = make_tensor(make_gmem_ptr(reinterpret_cast<float *>(params.rowmax_ptr) + offset_rowmax),
                            Shape<Int<kBlockM>>{},
                            make_stride(params.rowmax_row_stride));
    Tensor gdGK = make_tensor(make_gmem_ptr(reinterpret_cast<float *>(IsGating ? params.dlog_g_k_ptr : params.k_ptr) + offset_dgk),
                             Shape<Int<kBlockN>>{},
                             make_stride(params.dg_row_stride));
    Tensor gdGQ = make_tensor(make_gmem_ptr(reinterpret_cast<float *>(IsGating ? params.dlog_g_q_ptr : params.q_ptr) + offset_dgq),
                             Shape<Int<kBlockM>>{},
                             make_stride(params.dg_row_stride));

    Tensor gdQaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dq_accum_ptr) + offset_dq_accum),
                                  Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                  make_stride(params.dq_row_stride, _1{}));

    // if (thread0()) {
    //     printf("gQ.data(): %p\n", gQ.data());
    //     printf("gK.data(): %p\n", gK.data());
    //     printf("gV.data(): %p\n", gV.data());
    //     printf("gGQ.data(): %p\n", gGQ.data());
    //     printf("gGK.data(): %p\n", gGK.data());
    //     printf("gdGQ.data(): %p\n", gdGQ.data());
    //     printf("gdGK.data(): %p\n", gdGK.data());
    //     printf("gdQaccum.data(): %p\n", gdQaccum.data());
    //     printf("offset_gq: %d\n", offset_gq);
    //     printf("params.g_row_stride: %d\n", params.g_row_stride);
    //     printf("IsGating: %d\n", IsGating);
    // }
    ////////////////Shared Memory////////////////////
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQdY{});
    Tensor sQt = make_tensor(sQ.data(),
                            typename Kernel_traits::SmemLayoutQdYtransposed{});
    Tensor sQtNoSwizzle = make_tensor(sQ.data(),
                            typename Kernel_traits::SmemLayoutQdYtransposedNoSwizzle{});
    // Double buffer for sQ
    Tensor sdY = make_tensor(sQ.data() + (Double_buffer ? 2 : 1) * size(sQ),
                            typename Kernel_traits::SmemLayoutQdY{});
    Tensor sdYt = make_tensor(sdY.data(),
                             typename Kernel_traits::SmemLayoutQdYtransposed{});
    Tensor sdYtransposedNoSwizzle = make_tensor(sdY.data(),
                                                typename Kernel_traits::SmemLayoutQdYtransposedNoSwizzle{});
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(&(*sdY.data()) + size(sdY))),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sKt = make_tensor(sK.data(),
                            typename Kernel_traits::SmemLayoutKtransposed{});
    Tensor sKtNoSwizzle = make_tensor(sK.data(),
                            typename Kernel_traits::SmemLayoutKtransposedNoSwizzle{});
    Tensor sV = make_tensor(sK.data() + size(sK),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sdy = make_tensor(make_smem_ptr(reinterpret_cast<float *>(&(*sV.data()) + int(Kernel_traits::Is_V_in_regs ? 0 : size(sV)))),
                            typename Kernel_traits::SmemLayoutdy{});
    Tensor sdS = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(&(*sdy.data()) + size(sdy))),
                             typename Kernel_traits::SmemLayoutPdS{});
    Tensor sdSt = make_tensor(sdS.data(),
                            typename Kernel_traits::SmemLayoutPdStransposed{});
    Tensor sdStNoSwizzle = make_tensor(sdS.data(),
                            typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});
    Tensor sP = make_tensor(sdS.data() + size(sdS),
                            typename Kernel_traits::SmemLayoutPdS{});
    Tensor sPt = make_tensor(sP.data(),
                            typename Kernel_traits::SmemLayoutPdStransposed{});
    Tensor sPtNoSwizzle = make_tensor(sP.data(),
                            typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});
    Tensor srowmax = make_tensor(make_smem_ptr(reinterpret_cast<float *>(&(*sP.data()) + size(sP))),
                             typename Kernel_traits::SmemLayoutRowmax{});
    Tensor sGQ = make_tensor(make_smem_ptr(reinterpret_cast<float *>(&(*srowmax.data()) + size(srowmax))),
                             typename Kernel_traits::SmemLayoutGQ{});
    Tensor sGK = make_tensor(make_smem_ptr(reinterpret_cast<float *>(&(*sGQ.data()) + size(sGQ))),
                             typename Kernel_traits::SmemLayoutGK{});
    // be careful, we are reusing smem
    Tensor sdGK = make_tensor(sGK.data(), typename Kernel_traits::SmemLayoutGK{});
    // Tensor sdGQ = make_tensor(sdGK.data() + size(sdGK), typename Kernel_traits::SmemLayoutGQ{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopydy gmem_tiled_copy_dy;
    auto gmem_thr_copy_dy = gmem_tiled_copy_dy.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopyRowmax gmem_tiled_copy_rowmax;
    auto gmem_thr_copy_rowmax = gmem_tiled_copy_rowmax.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopyLogG gmem_tiled_copy_LogG;
    auto gmem_thr_copy_LogG = gmem_tiled_copy_LogG.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopydLogG gmem_tiled_copy_dLogG;
    auto gmem_thr_copy_dLogG = gmem_tiled_copy_dLogG.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopydQaccumAtomicAdd gmem_tiled_copy_dQaccumAtomicAdd;
    auto gmem_thr_copy_dQaccumAtomicAdd = gmem_tiled_copy_dQaccumAtomicAdd.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tdYgdY = gmem_thr_copy_QKV.partition_S(gdY);
    Tensor tdYsdY = gmem_thr_copy_QKV.partition_D(sdY);
    Tensor tdygdy = gmem_thr_copy_dy.partition_S(gdy);
    Tensor tdysdy = gmem_thr_copy_dy.partition_D(sdy);
    Tensor tRMgRM = gmem_thr_copy_rowmax.partition_S(growmax);
    Tensor tRMsdRM = gmem_thr_copy_rowmax.partition_D(srowmax);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
    Tensor tGQgGQ = gmem_thr_copy_LogG.partition_S(gGQ);  // (GCPY, GCPY_M)
    Tensor tGQsGQ = gmem_thr_copy_LogG.partition_D(sGQ);
    Tensor tGKgGK = gmem_thr_copy_LogG.partition_S(gGK);  // (GCPY, GCPY_N)
    Tensor tGKsGK = gmem_thr_copy_LogG.partition_D(sGK);
    Tensor tgdQaccum = gmem_thr_copy_dQaccumAtomicAdd.partition_D(gdQaccum);

    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_A(sQ);         // (MMA,MMA_N,MMA_K)
    Tensor tSrK = thr_mma_sdp.partition_fragment_B(sK);         // (MMA,MMA_N,MMA_K)
    Tensor tdPrdY = thr_mma_sdp.partition_fragment_A(sdY);      // (MMA,MMA_N,MMA_K)
    Tensor tdPrV = thr_mma_sdp.partition_fragment_B(sV);        // (MMA,MMA_N,MMA_K)

    typename Kernel_traits::TiledMmadKV tiled_mma_dkv;
    auto thr_mma_dkv = tiled_mma_dkv.get_thread_slice(tidx);

    typename Kernel_traits::TiledMmadQ tiled_mma_dq;
    auto thr_mma_dq = tiled_mma_dq.get_thread_slice(tidx);

    Tensor acc_dk = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});  // MMA, MMA_N, MMA_K
    Tensor acc_dv = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});  // MMA, MMA_N, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_QdY = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_QdY = smem_tiled_copy_QdY.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_QdY.partition_S(sQ);
    Tensor tdPsdY = smem_thr_copy_QdY.partition_S(sdY);

    // Rearrange dimension N to reduce write bank conflicts
    auto smem_tiled_copy_KV = make_tiled_copy_B_warpcontiguousN<MMA_N_SdP>(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_KV.partition_S(sK);
    Tensor tdPsV = smem_thr_copy_KV.partition_S(sV);

    // Partition sP and sdS to match the accumulator partitioning
    // This reduces bank conflicts when writing to sP (but not sdS)
    auto smem_tiled_copy_PdS = make_tiled_copy_C_warpcontiguousN<MMA_N_SdP>(typename Kernel_traits::SmemCopyAtomPdS{}, tiled_mma_sdp);
    auto smem_thr_copy_PdS = smem_tiled_copy_PdS.get_thread_slice(tidx);
    Tensor tPsP = smem_thr_copy_PdS.partition_D(sP);      // ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor tdSsdS = smem_thr_copy_PdS.partition_D(sdS);   // ((Atom,AtomNum),PIPE_M,PIPE_N)

    auto smem_tiled_copy_PdSt = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dkv);
    auto smem_thr_copy_PdSt = smem_tiled_copy_PdSt.get_thread_slice(tidx);
    Tensor tdVsPt = smem_thr_copy_PdSt.partition_S(sPt);
    Tensor tdKsdSt = smem_thr_copy_PdSt.partition_S(sdSt);

    auto smem_tiled_copy_dQdS = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_dq);
    auto smem_thr_copy_dQdS = smem_tiled_copy_dQdS.get_thread_slice(tidx);
    Tensor tdQsdS = smem_thr_copy_dQdS.partition_S(sdS);

    auto smem_tiled_copy_dQKt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dq);
    auto smem_thr_copy_dQKt = smem_tiled_copy_dQKt.get_thread_slice(tidx);
    Tensor tdQsKt = smem_thr_copy_dQKt.partition_S(sKt);

    auto smem_tiled_copy_QdYt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dkv);
    auto smem_thr_copy_QdYt = smem_tiled_copy_QdYt.get_thread_slice(tidx);
    Tensor tdVsdYt = smem_thr_copy_QdYt.partition_S(sdYt);
    Tensor tdKsQt = smem_thr_copy_QdYt.partition_S(sQt);


    int m_block_min = (!Is_causal)
        ? 0
        : std::max(0, (n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right) / kBlockM); // To account for unequal seqlen

    // Predicate tensors for QdY
    static_assert(size<0>(sQ) == size<0>(sdY), "sQ and sdY must have the same number of rows");
    static_assert(size<1>(sQ) == size<1>(sdY), "sQ and sdY must have the same number of columns");
    Tensor cQdY = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M, BLK_K) -> (blk_m, blk_k)
    Tensor tQdYcQdY = gmem_thr_copy_QKV.partition_D(cQdY);
    // Allocate predicate tensors for k
    Tensor tQdYpQdY = make_tensor<bool>(make_shape(size<2>(tSsQ)));
    if constexpr (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQdYpQdY); ++k) { tQdYpQdY(k) = get<1>(tQdYcQdY(0, 0, k)) < params.d; }
    }

    // Predicate tensors for KV
    static_assert(size<0>(sV) == size<0>(sK), "sV and sK must have the same number of rows");
    static_assert(size<1>(sV) == size<1>(sK), "sV and sK must have the same number of columns");
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sV), size<1>(sV)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_D(cKV);
    // Allocate predicate tensors for k
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tVsV)));
    if constexpr(!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Loaders and bumpers, we can remove the if else with templated lambdas
    // when libtorch is compatible with c++20
    auto load_GQ = [&](const int m_block, const bool Clear_OOB_MN=true) {
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
        tGQgGQ.data() = tGQgGQ.data() + index_t(-kBlockM * params.g_row_stride);
    };
    auto load_GK = [&](bool Clear_OOB_MN=true) {
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
    };
    auto save_dGK = [&](bool Is_even=true) {
        if constexpr (!IsGating) {
            return;
        }
        Tensor cGK = make_identity_tensor(make_shape(size<0>(sGK)));
        Tensor tGKcGK = gmem_thr_copy_dLogG.partition_D(cGK);
        Tensor tdGKsdGK = gmem_thr_copy_dLogG.partition_S(sdGK);
        Tensor tdGKgdGK = gmem_thr_copy_dLogG.partition_D(gdGK);
        if constexpr (kNThreads <= kBlockN) {
            power::copy1d<Is_even_MN, false, kBlockN>(gmem_tiled_copy_dLogG, tdGKsdGK, tdGKgdGK, tGKcGK, binfo.actual_seqlen_k - n_block * kBlockN);
        } else if (tidx < kBlockN) {
            power::copy1d<Is_even_MN, false, kBlockN>(gmem_tiled_copy_dLogG, tdGKsdGK, tdGKgdGK, tGKcGK, binfo.actual_seqlen_k - n_block * kBlockN);
        }
    };
    auto load_dY = [&](const int m_block, const bool Is_even=true, const bool Clear_OOB_MN=true) {
        if (!Is_even || !Is_even_K) { // if we are handling the last block of dY or k is not divisible by 16
            if (Clear_OOB_MN) {
                power::copy<Is_even_MN, Is_even_K, true, kBlockM>(gmem_tiled_copy_QKV, tdYgdY, tdYsdY, tQdYcQdY, tQdYpQdY, binfo.actual_seqlen_q - m_block * kBlockM);
            } else {
                power::copy<Is_even_MN, Is_even_K, false, kBlockM>(gmem_tiled_copy_QKV, tdYgdY, tdYsdY, tQdYcQdY, tQdYpQdY, binfo.actual_seqlen_q - m_block * kBlockM);
            }
        } else { // nice even copy
            cute::copy(gmem_tiled_copy_QKV, tdYgdY, tdYsdY);
        }
        tdYgdY.data() = tdYgdY.data() + index_t(-kBlockM * params.dY_row_stride);
    };
    auto load_Q = [&](const int m_block, const bool Is_even_round=true, const bool Clear_OOB_MN=true) {
        if (!Is_even_round || !Is_even_K) { // if we are handling the last block of Q or K is not divisible by 16
            if (Clear_OOB_MN) {
                power::copy<Is_even_MN, Is_even_K, true, kBlockM>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQdYcQdY, tQdYpQdY, binfo.actual_seqlen_q - m_block * kBlockM);
            } else {
                power::copy<Is_even_MN, Is_even_K, false, kBlockM>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQdYcQdY, tQdYpQdY, binfo.actual_seqlen_q - m_block * kBlockM);
            }
        } else { // nice even copy
            cute::copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);
        }
        tQgQ.data() = tQgQ.data() + index_t(-kBlockM * params.q_row_stride);
    };
    auto load_K = [&](bool Is_even=true, bool Clear_OOB_MN=true) {
        if (!Is_even || !Is_even_K) { // if we are handling the last block of Q or K is not divisible by 16
            if (Clear_OOB_MN) {
                power::copy<Is_even_MN, Is_even_K, true, kBlockN>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN);
            } else {
                power::copy<Is_even_MN, Is_even_K, false, kBlockN>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN);
            }
        } else { // nice even copy
            cute::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
        }
    };
    auto load_V = [&](bool Is_even=false, bool Clear_OOB_MN=true) {
        if (!Is_even || !Is_even_K) { // if we are handling the last block of Q or K is not divisible by 16
            if (Clear_OOB_MN) {
                power::copy<Is_even_MN, Is_even_K, true, kBlockN>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN);
            } else {
                power::copy<Is_even_MN, Is_even_K, false, kBlockN>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN);
            }
        } else { // nice even copy
            cute::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
        }
    };

    // debugger
    #define DEBUG_PRINT(acc, name) \
        if (true) { \
            auto r_acc = make_tensor(acc.data(), acc.layout()); \
            Tensor t_acc = smem_thr_copy_PdS.retile_S(r_acc); \
            Tensor t_acc_converted = power::convert_type<Element>(t_acc); \
            cute::copy(smem_tiled_copy_PdS, t_acc_converted, tdSsdS); \
            __syncthreads(); \
            if (DEBUGGER_THREAD) { \
                printf("dKdVdQ KERNEL: m_block = %d, %s:\n", m_block, name); \
                print_tensor(sdS); \
                printf("\n"); \
            } \
            __syncthreads(); \
        }

    // Prologue
    // Register for masking
    Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});    // (BLK_M,BLK_N) -> (blk_m,blk_n)
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);                           // (MMA,MMA_N,MMA_N)
    static_assert(decltype(size<0>(taccScS))::value == 4);
    // Convert to ((2, 2), MMA_M, MMA_N) then take only the row indices.
    Tensor taccScS_row = logical_divide(taccScS, Shape<_2>{})(make_coord(0, _), _, 0);
    Tensor r_rowmax = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    CUTE_UNROLL
    for (int mi = 0; mi < size(r_rowmax); ++mi) {
        const int row = get<0>(taccScS_row(mi));
        r_rowmax(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? growmax(row) : INFINITY;
    }
    Tensor rdy = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    CUTE_UNROLL
    for (int mi = 0; mi < size(rdy); ++mi) {
        const int row = get<0>(taccScS_row(mi));
        rdy(mi) = row < binfo.actual_seqlen_q - m_block * kBlockM ? gdy(row) : 0.0f;
    }


    // Initialize double buffer position for sQ
    if (Double_buffer && m_block % 2 == 1) { 
        tQsQ.data() = tQsQ.data() + size(sQ);
        tSsQ.data() = tSsQ.data() + size(sQ);
        tdKsQt.data() = tdKsQt.data() + size(sQ);
    }

    // load K, V, GK
    if (Kernel_traits::Is_V_in_regs) {
        load_V(!is_last_n_block);
        cute::cp_async_fence();
    }

    load_K(!is_last_n_block);
    load_GK();

    if (Kernel_traits::Is_V_in_regs) { // optionally load V into registers
        cute::cp_async_wait<0>();
        __syncthreads();
        Tensor tdPrV_copy_view = smem_thr_copy_KV.retile_D(tdPrV);
        CUTE_STATIC_ASSERT_V(size<1>(tdPsV) == size<1>(tdPrV_copy_view));            // M
        cute::copy(smem_tiled_copy_KV, tdPsV, tdPrV_copy_view);
        __syncthreads();
    } else {
        load_V(!is_last_n_block);
    }
    cute::cp_async_fence();

    // load G_Q, G_K
    load_Q(m_block, false); // start from last Q block, so OOB possible
    if constexpr (Double_buffer) {
        tQsQ.data() = tQsQ.data() + (m_block % 2 == 0 ? size(sQ) : -size(sQ));
    }
    load_GQ(m_block);
    cute::cp_async_fence();

    // Load dY
    load_dY(m_block, false);
    cute::cp_async_fence();

    // clear accumulators
    clear(acc_dk);
    clear(acc_dv);

    // Registers for storing dGK
    using acc_tensor = decltype(partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{}));
    using acc_layout = decltype(acc_tensor{}.layout());
    using rowcol_layout = decltype(power::convert_layout_acc_rowcol(acc_layout{}));
    Tensor acc_dGK = make_tensor<float>(Shape<Int<(IsGating ? size<1>(rowcol_layout{}) : 0)>>{});
    clear(acc_dGK);

    for (; m_block >= m_block_min; --m_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_s);
        // makes sure Q, GQ, rowmax are loaded
        power_attention::cp_async_wait<Double_buffer ? 1 : 0>();
        __syncthreads();

#ifdef DEBUG_POWER_BWD_DKDVDQ
        if (DEBUGGER_THREAD) {
            printf("m_block: %d, srowmax: \n");
            print_tensor(srowmax);
            printf("\n");
            printf("m_block: %d sGQ:\n", m_block);
            print_tensor(sGQ);
            printf("\n");
            printf("m_block: %d sQ:\n", m_block);
            print_tensor(sQ);
            printf("\n");
        }
#endif

        // start loading Q[i+1]
        if (Double_buffer && m_block > m_block_min) {
            load_Q(m_block - 1, false);
            tQsQ.data() = tQsQ.data() + (m_block % 2 == 1 ? size(sQ) : -size(sQ));
        }

        // compute S = QK^T
        power::gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma_sdp,
                    smem_tiled_copy_QdY, smem_tiled_copy_KV, smem_thr_copy_QdY, smem_thr_copy_KV);

        // bump tSsQ, we are done with it
        if (Double_buffer && m_block > m_block_min) {
            tSsQ.data() = tSsQ.data() + (m_block % 2 == 1 ? -size(sQ) : size(sQ));
        }

        // log S
#ifdef DEBUG_POWER_BWD_DKDVDQ
        DEBUG_PRINT(acc_s, "S = QK^T")
#endif
        
        // Reshape acc_s from (MMA=4, MMA_N, MMA_N) to (row=(2, MMA_N), col=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), power::convert_layout_acc_rowcol(acc_s.layout()));
        Tensor signs = make_tensor<bool>(scores.layout());

        CUTE_UNROLL
        for(int i = 0; i < size(scores); i++) {
            signs(i) = scores(i) >= 0;
        }

        auto taccsS = thr_mma_sdp.partition_C(sdS);
        cute::copy(acc_s, taccsS);

        // Apply mask, get Z[i] * M
        if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k
            || (!Is_even_MN && ((n_block + 1) * kBlockN >= binfo.actual_seqlen_k) || (m_block + 1) * kBlockM >= binfo.actual_seqlen_q)) {
            power::apply_mask_causal<AtomLayoutMS * 16, !NormalSpace>(scores, n_block * kBlockN + (tidx / 32 / AtomLayoutMS) * MMA_N_SdP * 16,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q);
        }

#ifdef DEBUG_POWER_BWD_DKDVDQ
        DEBUG_PRINT(acc_s, "after mask")
#endif

        // Logspace: T = p * log(abs(S) + ε) - log_stabilizer
        // Normalspace || FlashEquivalent: T = S / stabilizer
        if constexpr (FlashEquivalent || NormalSpace) {
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

#ifdef DEBUG_POWER_BWD_DKDVDQ
        DEBUG_PRINT(acc_s, "after abslogp")
        if (DEBUGGER_THREAD) {
            printf("m_block = %d, sGQ:\n", m_block);
            print_tensor(sGQ);
            printf("\n");
            printf("m_block = %d, sGK:\n", m_block);
            print_tensor(sGK);
            printf("\n");
        }
#endif

        // Apply gating
        if constexpr (IsGating) {
            // !NormalSpace: Z = Z + (G_Q @ 1^T - 1 @ G_K^T)
            // NormalSpace: Z = Z * (G_Q @ 1^T - 1 @ G_K^T)^{1/p}
            power::apply_gating</*masked=*/true, /*logspace=*/!NormalSpace, typename Kernel_traits::TiledMmaSdP, MMA_N_SdP>(acc_s, sGQ, sGK, Deg);
        }

#ifdef DEBUG_POWER_BWD_DKDVDQ
        DEBUG_PRINT(acc_s, "after gating")
#endif

        // Apply exp
        if constexpr (NormalSpace) {
            // NormalSpace: P = (Z / Z_rowmax)^p
            power::scale_apply_power<Deg>(scores, r_rowmax);
        } else {
            // !NormalSpace: P = exp(Z * M)
            power::scale_apply_exp(scores, r_rowmax);
            if constexpr (Deg % 2 == 1) {
                power::template apply_signs(scores, signs);
            }
        }

#ifdef DEBUG_POWER_BWD_DKDVDQ
        DEBUG_PRINT(acc_s, "P after exp")
        __syncthreads();
        if (DEBUGGER_THREAD) {
            printf("m_block = %d, srowmax:\n", m_block);
            print_tensor(srowmax);
            printf("\n");
            printf("m_block = %d, r_rowmax:\n", m_block);
            CUTE_UNROLL
            for (int i = 0; i < size(r_rowmax); ++i) {
                printf("%f ", r_rowmax[i]);
            }
            printf("\n");
        }
        __syncthreads();
#endif

        // Convert P[i] to fp16/bf16
        Tensor rP = power::convert_type<Element>(acc_s);
        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_N, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_N, MMA_N) if using m16n8k8.
        Tensor tPrP = make_tensor(rP.data(), power::convert_layout_acc_Aregs<Kernel_traits::TiledMmaSdP>(rP.layout()));
        Tensor tPaP = smem_thr_copy_PdS.retile_S(tPrP);     // ((Atom,AtomNum), MMA_N, MMA_N)

        // wait for dY and dy
        power::cp_async_wait<0>();
        __syncthreads();
        if (m_block > m_block_min) { // load next
            // load_rowmax(m_block - 1);
            if constexpr (IsGating) {
                load_GQ(m_block - 1);
            }
        }
        cute::cp_async_fence();
        // put P[i] back to smem
        cute::copy(smem_tiled_copy_PdS, tPaP, tPsP);

        // Starting computing dP[i] = dY[i] @ V + dy[i] @ 1^T
        Tensor acc_dp = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        CUTE_STATIC_ASSERT_V(size<0>(acc_dp) == size<0>(acc_s));                     // MMA
        CUTE_STATIC_ASSERT_V(size<1>(acc_dp) == size<1>(acc_s));                     // MMA
        CUTE_STATIC_ASSERT_V(size<2>(acc_dp) == size<2>(acc_s));                     // MMA
        clear(acc_dp);

#ifdef DEBUG_POWER_BWD_DKDVDQ
        __syncthreads();
        if (DEBUGGER_THREAD) {
            printf("m_block = %d, sdY: \n", m_block);
            print_tensor(sdY);
            printf("\n");
            printf("m_block = %d, sV: \n", m_block);
            print_tensor(sV);
            printf("\n");
        }
        __syncthreads();
#endif

        // // read in dy from smem
        // Tensor rdy = read_row_summary<size<1>(acc_dp) * 2, Kernel_traits::TiledMmaSdP>(sdy);

        // Compute dP[i] = dY[i] @ V
        power::gemm</*A_in_regs=*/false, /*B_in_regs=*/Kernel_traits::Is_V_in_regs>(
            acc_dp, tdPrdY, tdPrV, tdPsdY, tdPsV, tiled_mma_sdp,
            smem_tiled_copy_QdY, smem_tiled_copy_KV, smem_thr_copy_QdY, smem_thr_copy_KV
        );

#ifdef DEBUG_POWER_BWD_DKDVDQ
        DEBUG_PRINT(acc_dp, "dP = dY @ V")
        if (DEBUGGER_THREAD) {
            printf("m_block = %d, sdy:\n", m_block);
            print_tensor(sdy);
            printf("\n");
            printf("m_block = %d, rdy:\n", m_block);
            CUTE_UNROLL
            for (int i = 0; i < size(rdy); ++i) {
                printf("%f ", rdy[i]);
            }
            printf("\n");
        }
        __syncthreads();
#endif

        // Reshape acc_dp from (MMA=4, MMA_N, MMA_N) to (row=(2, MMA_N), col=(2, MMA_N))
        Tensor dZ = make_tensor(acc_dp.data(), scores.layout());

        // Compute dZ, or dT
        CUTE_UNROLL
        for (int mi = 0; mi < size<0>(dZ); ++mi) {
            CUTE_UNROLL
            for (int ni = 0; ni < size<1>(dZ); ++ni) {
                dZ(mi, ni) += rdy(mi); // add dy, dP = dY @ V + dy @ 1^T
                dZ(mi, ni) = dZ(mi, ni) * scores(mi, ni); // dZ = dP * P
            }
        }

#ifdef DEBUG_POWER_BWD_DKDVDQ
        DEBUG_PRINT(acc_dp, "dZ");
#endif

        // get gradients for G_Q and G_K
        if constexpr (IsGating) {
            Tensor acc_dGQ = make_tensor<ElementAccum>(Shape<Int<size<0>(rowcol_layout{})>>{});
            clear(acc_dGQ);

            CUTE_UNROLL
            for (int ni = 0; ni < size<1>(dZ); ++ni) {
                CUTE_UNROLL
                for (int mi = 0; mi < size<0>(dZ); ++mi) {
                    acc_dGK(ni) -= dZ(mi, ni);
                    acc_dGQ(mi) += dZ(mi, ni);
                }
            }


            SumOp<float> sum_op;
            quad_allreduce_(acc_dGQ, acc_dGQ, sum_op); // sum over 4 adjacent threads

            if constexpr (AtomLayoutNSdP > 1) {
                constexpr static int AtomShape_M = size<0>(typename Kernel_traits::TiledMmaSdP::Shape_MNK{});
                constexpr static int AtomLayoutM = size<1>(typename Kernel_traits::TiledMmaSdP::ThrLayoutVMNK{});
                constexpr static int m_tile_stride = AtomLayoutM * AtomShape_M;
                const int offset_base = ((threadIdx.x / 32) % AtomLayoutM) * AtomShape_M + (threadIdx.x % 32) / 4;

                if (tidx % 4 == 0) {
                    CUTE_UNROLL
                    for (int mi = 0; mi < size(acc_dGQ); ++mi) {
                        const int m = (mi / 2) * m_tile_stride + offset_base + (mi % 2) * (AtomShape_M / 2);
                        if (m < std::min(binfo.actual_seqlen_q - m_block * kBlockM, kBlockM)) {
                            atomicAdd(&gdGQ(m), acc_dGQ(mi));
                        }
                    }
                }
#ifdef DEBUG_POWER_BWD_DKDVDQ
                __syncthreads();
                if (DEBUGGER_THREAD) {
                    printf("thread 0 gdGQ: ");
                    for (int i = 0; i < size(gdGQ); ++i) {
                        printf("%f ", gdGQ[i]);
                    }
                    printf("\n");
                }
                __syncthreads();
#endif
                gdGQ.data() = gdGQ.data() + index_t(-kBlockM * params.dg_row_stride);
            }

        }

        // Load original
        Tensor acc_rS = make_tensor_like(acc_dp);
        Tensor taccsS_origin = thr_mma_sdp.partition_C(sdS);
        cute::copy(taccsS_origin, acc_rS);


        // dZ*p
        if constexpr (!FlashEquivalent) {
            power::template multiply_by_deg<Deg>(dZ);
        }

#ifdef DEBUG_POWER_BWD_DKDVDQ
        DEBUG_PRINT(acc_dp, "dZ * p");
#endif


        // Compute dS = dT * p * sign(S) / (abs(S) + ε)
        Tensor dS = make_tensor(dZ.data(), dZ.layout());
        Tensor S_rowcol = make_tensor(acc_rS.data(), scores.layout());
        if constexpr (!FlashEquivalent) {
            CUTE_UNROLL
            for (int i = 0; i < size(dZ); ++i) {
                dS(i) = (S_rowcol(i) < 0 ? -dZ(i) : dZ(i)) / (cuda_abs(S_rowcol(i)) + params.ε); 
            }
        } else {
            CUTE_UNROLL
            for (int i = 0; i < size(dZ); ++i) {
                dS(i) = dZ(i) / params.stabilizer;
            }
        }


        // copy dS to smem
        Tensor dS_reshaped = make_tensor(dS.data(), acc_dp.layout());
        // Convert dS from fp32 to fp16
        Tensor tdSrdS = power::convert_type<Element>(dS_reshaped);
        // if (cute::thread0()) { print(tPrP); }
        Tensor tdSadS = smem_thr_copy_PdS.retile_S(tdSrdS);       // ((Atom,AtomNum), MMA_N, MMA_N)
        __syncthreads(); // make sure P is written to smem
        // start loading dY[i+1]
        cute::copy(smem_tiled_copy_PdS, tdSadS, tdSsdS);

#ifdef DEBUG_POWER_BWD_DKDVDQ
        if (DEBUGGER_THREAD) {
            printf("m_block = %d, acc_dv before gemm:\n", m_block);
            for (int i = 0; i < size(acc_dv); ++i) {
                printf("%f ", acc_dv[i]);
            }
            printf("\n");
        }
#endif

        // compute dV[i] = P^T @ dY
        Tensor tdVrPt = thr_mma_dkv.partition_fragment_A(sPtNoSwizzle);   // (MMA, MMA_N, MMA_N)
        Tensor tdVrdY = thr_mma_dkv.partition_fragment_B(sdYtransposedNoSwizzle); // (MMA, MMA_K, MMA_N)
        power::gemm(acc_dv, tdVrPt, tdVrdY, tdVsPt, tdVsdYt, tiled_mma_dkv,
                    smem_tiled_copy_PdSt, smem_tiled_copy_QdYt, smem_thr_copy_PdSt, smem_thr_copy_QdYt);

#ifdef DEBUG_POWER_BWD_DKDVDQ
        if (DEBUGGER_THREAD) {
            printf("m_block = %d, acc_dv after gemm:\n", m_block);
            for (int i = 0; i < size(acc_dv); ++i) {
                printf("%f ", acc_dv[i]);
            }
            printf("\n");
        }
#endif

        // start loading dY[i+1], dS is written to smem
        __syncthreads();
        if (m_block > m_block_min) {
            load_dY(m_block - 1, false);
        }
        cute::cp_async_fence();


#ifdef DEBUG_POWER_BWD_DKDVDQ
        if (DEBUGGER_THREAD) {
            printf("m_block = %d, acc_dGK:\n", m_block);
            for (int i = 0; i < size(acc_dGK); ++i) {
                printf("%f ", acc_dGK[i]);
            }
            printf("\n");
        }
        __syncthreads();
#endif


#ifdef DEBUG_POWER_BWD_DKDVDQ
        DEBUG_PRINT(acc_dp, "dS")
#endif
        // Compute dQ[i] = dS @ K
        Tensor acc_dq = partition_fragment_C(tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // (MMA, MMA_N, MMA_N)
        Tensor tdQrdS = thr_mma_dq.partition_fragment_A(sdS);
        Tensor tdQrKt = thr_mma_dq.partition_fragment_B(sKtNoSwizzle); // can we afford to keep this in regs?
        clear(acc_dq);
        power::gemm(acc_dq, tdQrdS, tdQrKt, tdQsdS, tdQsKt, tiled_mma_dq, smem_tiled_copy_dQdS, smem_tiled_copy_dQKt, smem_thr_copy_dQdS, smem_thr_copy_dQKt);

        if (m_block > m_block_min) { // load next rowmax
            growmax.data() = growmax.data() + (-int(kBlockM));
            CUTE_UNROLL
            for (int mi = 0; mi < size(r_rowmax); ++mi) { r_rowmax(mi) = growmax(get<0>(taccScS_row(mi))); }

            gdy.data() = gdy.data() + index_t(-kBlockM * params.dy_row_stride);
            CUTE_UNROLL
            for (int mi = 0; mi < size(rdy); ++mi) { rdy(mi) = gdy(get<0>(taccScS_row(mi))); }
        }

        // atomicAdd 
        CUTE_UNROLL
        for (int i = 0; i < size(acc_dq); ++i) {
            atomicAdd(&tgdQaccum(i), acc_dq(i));
        }
        tgdQaccum.data() = tgdQaccum.data() + index_t(-kBlockM * params.dq_row_stride);

        // Compute dK[i] = dS^T @ Q
        Tensor tdKrdSt = thr_mma_dkv.partition_fragment_A(sdStNoSwizzle); // (MMA, MMA_N, MMA_N)
        Tensor tdKrQt = thr_mma_dkv.partition_fragment_B(sQtNoSwizzle);   // (MMA, MMA_K, MMA_N)
        power::gemm(acc_dk, tdKrdSt, tdKrQt, tdKsdSt, tdKsQt, tiled_mma_dkv,
                    smem_tiled_copy_PdSt, smem_tiled_copy_QdYt, smem_thr_copy_PdSt, smem_thr_copy_QdYt);

        // Bump tdKsQt, we are done with it
        if (Double_buffer && m_block > m_block_min) {
            tdKsQt.data() = tdKsQt.data() + (m_block % 2 == 0 ? size(sQ) : -size(sQ));
        }

        // if not double buffer, this is where we start loading next Q
        if (!Double_buffer && m_block > m_block_min) {
            __syncthreads();
            load_Q(m_block - 1, false);
            cute::cp_async_fence();
        }
    }

    // Epilogue
    __syncthreads();

    // Sum acc_dGK
    if constexpr (IsGating) {
        SumOp<float> sum_op;
        power_attention::quad_allreduce_mod_(acc_dGK, acc_dGK, sum_op); // sum over threads that have the same mod 4
        const int warp_id = tidx / 32;
        constexpr int warp_group_max = Kernel_traits::AtomLayoutMSdP - 1;
        // warp reduce
        for (int warp_group = warp_group_max; warp_group >= 0; --warp_group) {
            if (warp_id % Kernel_traits::AtomLayoutMSdP == warp_group) {
                if ((tidx % 32) / 4 == 0) {
                    for (int ni = 0; ni < size(acc_dGK); ++ni) {
                        int n = ni_to_n_warpNcontiguous<Kernel_traits::TiledMmaSdP, MMA_N_SdP>(ni);
                        sdGK[n] = warp_group == warp_group_max ? acc_dGK(ni) : acc_dGK(ni) + sdGK[n];
                    }
                }
            }
            __syncthreads();
        }
        // write dGK back to global memory
        save_dGK();
    }

#ifdef DEBUG_POWER_BWD_DKDVDQ
    if (DEBUGGER_THREAD) {
        printf("acc_dk:\n");
        for (int i = 0; i < size(acc_dk); ++i) {
            printf("%f ", acc_dk[i]);
        }
        printf("\n");
        printf("acc_dv:\n");
        for (int i = 0; i < size(acc_dv); ++i) {
            printf("%f ", acc_dv[i]);
        }
        printf("\n");
    }
#endif
    // Convert acc_dv from fp32 to fp16
    Tensor rdK = make_tensor<Element>(acc_dk.layout());
    Tensor rdV = make_tensor<Element>(acc_dv.layout());
    power::convert_type(acc_dk, rdK);
    power::convert_type(acc_dv, rdV);

#ifdef DEBUG_POWER_BWD_DKDVDQ
    if (DEBUGGER_THREAD) {
        printf("rdK:\n");
        for (int i = 0; i < size(rdK); ++i) {
            printf("%f ", rdK[i]);
        }
        printf("\n");
        printf("rdV:\n");
        for (int i = 0; i < size(rdV); ++i) {
            printf("%f ", rdV[i]);
        }
        printf("\n");
    }
#endif

    Tensor sdK = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutdKV{});  // (SMEM_N, SMEM_K)
    Tensor sdV = make_tensor(sdK.data() + size(sdK), typename Kernel_traits::SmemLayoutdKV{}); // (SMEM_N, SMEM_K)

    // Partition sdV and sdK to match the accumulator partitioning
    auto smem_tiled_copy_dKV = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdKV{}, tiled_mma_dkv);
    auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(rdK);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(sdK);   // ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(rdV);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(sdV);    // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // We need syncthreads here since we're writing to the same location as sK and sV.
    // Without syncthreads, some thread might modify the location of sK while another thread
    // is reading it for dQ gemm, leading to a race condition.
    // If Is_last, there's already a __syncthreads() at the end of the loop.
    __syncthreads();


    // copy dK and dV back to smem
    cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);
    cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);

    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
       + n_block * kBlockN * params.dk_row_stride + (bidh / params.h_h_k_ratio) * params.dk_head_stride;
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
       + n_block * kBlockN * params.dv_row_stride + (bidh / params.h_h_k_ratio) * params.dv_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                             Shape<Int<kBlockN>, Int<kHeadDim>>{},
                             make_stride(params.dk_row_stride, _1{}));
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                             Shape<Int<kBlockN>, Int<kHeadDim>>{},
                             make_stride(params.dv_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
    auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor tdKsdK = gmem_thr_copy_dKV.partition_S(sdK);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
    Tensor tdVsdV = gmem_thr_copy_dKV.partition_S(sdV);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);

    __syncthreads();

#ifdef DEBUG_POWER_BWD_DKDVDQ
    if (DEBUGGER_THREAD) {
        printf("sdK:\n");
        print_tensor(sdK);
        printf("\n");
        printf("sdV:\n");
        print_tensor(sdV);
        printf("\n");
    }
#endif

    // copy dK and dV back to gmem
    Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
    cute::copy(gmem_tiled_copy_dKV, tdKsdK, tdKrdK);
    Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
    cute::copy(gmem_tiled_copy_dKV, tdVsdV, tdVrdV);
    Tensor cdKV = make_identity_tensor(make_shape(size<0>(sdK), size<1>(sdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
    Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
    #pragma unroll
    for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d; }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    power::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    power::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );

}

////////////////////////////////////////////////////////////////////////////////////////////////////

// template<typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool IsGating, typename Params>
// inline __device__ void compute_dk_dv(const Params &params) {

//     // The block index for the batch.
//     const int bidb = blockIdx.y;
//     // The block index for the head.
//     const int bidh = blockIdx.z;
//     // The block index for the sequence.
//     const int n_block = blockIdx.x;

//     compute_dk_dv_1colblock<Kernel_traits, Is_causal, Is_even_MN, Is_even_K, IsGating>(params, bidb, bidh, n_block);
// }


// template<typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool IsGating, typename Params>
// inline __device__ void compute_dq(const Params &params) {

//     // The block index for the batch.
//     const int bidb = blockIdx.y;
//     // The block index for the head.
//     const int bidh = blockIdx.z;
//     // The block index for the sequence.
//     const int m_block = blockIdx.x;

//     compute_dq_1rowblock<Kernel_traits, Is_causal, Is_even_MN, Is_even_K, IsGating>(params, bidb, bidh, m_block);
// }

template<typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool IsGating, bool FlashEquivalent, bool NormalSpace, int Deg, typename Params>
inline __device__ void compute_dq_dk_dv(const Params &params) {

    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;

    // Each CTA process a number of k blocks determined b the grid size
    for (int n_block = blockIdx.x; n_block < (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN; n_block += gridDim.x) {
        compute_dq_dk_dv_1colblock<Kernel_traits, Is_causal, Is_even_MN, Is_even_K, IsGating, FlashEquivalent, NormalSpace, Deg>(params, bidb, bidh, n_block);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace power
