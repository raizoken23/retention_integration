/******************************************************************************
 * Copyright (c) 2024, Sean Zhang.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>
#include "utils.h"

namespace power {

using namespace cute;

template <typename TiledMma>
__forceinline__ __device__ int mi_to_m(const int mi) {
    constexpr static int AtomShape_M = size<0>(typename TiledMma::Shape_MNK{});
    constexpr static int AtomLayoutM = size<1>(typename TiledMma::ThrLayoutVMNK{});
    constexpr static int m_tile_stride = AtomLayoutM * AtomShape_M;
    return (mi / 2) * m_tile_stride + ((threadIdx.x / 32) % AtomLayoutM) * AtomShape_M + (threadIdx.x % 32) / 4 + (mi % 2) * (AtomShape_M / 2);
    // return 0;
}


template <typename TiledMma>
__forceinline__ __device__ int ni_to_n(const int ni) {
    constexpr static int AtomShape_N = size<1>(typename TiledMma::Shape_MNK{});
    constexpr static int AtomLayoutM = size<1>(typename TiledMma::ThrLayoutVMNK{});
    constexpr static int AtomLayoutN = size<2>(typename TiledMma::ThrLayoutVMNK{});
    constexpr static int n_tile_stride = AtomLayoutN * AtomShape_N;
    return (ni / 2) * n_tile_stride + ((threadIdx.x / 32) / AtomLayoutM) * AtomShape_N + (threadIdx.x % 4) * 2 + (ni % 2);
    // return 0;
}


template <typename TiledMma, int MMA_N>
__forceinline__ __device__ int ni_to_n_warpNcontiguous(const int ni) {
    constexpr static int AtomShape_N = size<1>(typename TiledMma::Shape_MNK{});
    constexpr static int AtomLayoutM = size<1>(typename TiledMma::ThrLayoutVMNK{});
    constexpr static int AtomLayoutN = size<2>(typename TiledMma::ThrLayoutVMNK{});
    constexpr static int n_tile_small_stride = AtomShape_N;
    constexpr static int n_tile_large_stride = MMA_N * AtomShape_N * 2;
    return ((threadIdx.x / 32) / AtomLayoutM) * n_tile_large_stride + (ni / 2) * n_tile_small_stride + (threadIdx.x % 4) * 2 + (ni % 2);
    // return 0;
}


template <bool masked=true, bool logspace=true, typename TensorS, typename TensorSRG, typename TensorSCG>
__forceinline__ __device__ void apply_gating_rowcol(TensorS &scores_rowcol, TensorSRG &rGQ, TensorSCG &rGK, const int deg) {
    CUTE_UNROLL
    for (int mi = 0; mi < size<0>(scores_rowcol); ++mi) {
        for (int ni = 0; ni < size<1>(scores_rowcol); ++ni) {
            auto discount = rGQ[mi] - rGK[ni];
            if constexpr (logspace) {
                if constexpr (masked) {
                    scores_rowcol(mi, ni) = scores_rowcol(mi, ni) == -INFINITY ? -INFINITY : discount + scores_rowcol(mi, ni);
                }
                else {
                    scores_rowcol(mi, ni) += discount;
                }
            } else {
                if constexpr (masked) {
                    scores_rowcol(mi, ni) = scores_rowcol(mi, ni) == 0.0f ? 0.0f :  scores_rowcol(mi, ni) * expf(discount / deg);
                }
                else {
                    scores_rowcol(mi, ni) = scores_rowcol(mi, ni) * expf(discount / deg);
                }
            }
        }
    }
}


template <bool masked=true, bool logspace=true, typename TiledMma, int MMA_N=-1, typename TensorS, typename TensorSRG, typename TensorSCG>
__forceinline__ __device__ void apply_gating(TensorS &acc_s, TensorSRG &row_gating, TensorSCG &col_gating, const int deg) {
    
    Tensor scores_rowcol = make_tensor(acc_s.data(), power::convert_layout_acc_rowcol(acc_s.layout()));

    Tensor rGQ = make_tensor<float>(Shape<Int<size<0>(scores_rowcol)>>{});
    Tensor rGK = make_tensor<float>(Shape<Int<size<1>(scores_rowcol)>>{});

    CUTE_UNROLL
    for (int mi = 0; mi < size<0>(scores_rowcol); ++mi) {
        rGQ[mi] = row_gating(mi_to_m<TiledMma>(mi));
    }
    
    if constexpr (MMA_N == -1) {
        CUTE_UNROLL
        for (int ni = 0; ni < size<1>(scores_rowcol); ++ni) {
            rGK[ni] = col_gating(ni_to_n<TiledMma>(ni));
        }
    } else if constexpr (MMA_N > 0) {
        CUTE_UNROLL
        for (int ni = 0; ni < size<1>(scores_rowcol); ++ni) {
            rGK[ni] = col_gating(ni_to_n_warpNcontiguous<TiledMma, MMA_N>(ni));
        }
    }

    apply_gating_rowcol<masked, logspace>(scores_rowcol, rGQ, rGK, deg);
}


template <bool masked=true, typename TiledMma, int MMA_N=-1, typename TensorS, typename TensorRGQ, typename TensorSGK>
__forceinline__ __device__ void apply_gating_with_q(TensorS &acc_s, TensorRGQ &rGQ, TensorSGK &col_gating) {
    
    Tensor scores_rowcol = make_tensor(acc_s.data(), power::convert_layout_acc_rowcol(acc_s.layout()));

    Tensor rGK = make_tensor<float>(Shape<Int<size<1>(scores_rowcol)>>{});
    
    CUTE_UNROLL
    for (int ni = 0; ni < size<1>(scores_rowcol); ++ni) {
        rGK[ni] = col_gating(ni_to_n_warpNcontiguous<TiledMma, MMA_N>(ni));
    }

    apply_gating_rowcol<masked>(scores_rowcol, rGQ, rGK);
}


} // namespace power
