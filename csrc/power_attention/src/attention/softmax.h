/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/numeric_types.h>

#include "utils.h"

namespace power {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            summary(mi) = op(summary(mi), tensor(mi, ni));
        }
    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    thread_reduce_<zero_init>(tensor, summary, op);
    quad_allreduce_(summary, summary, op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max){
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_absmax(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max){
    AbsMaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
    SumOp<float> sum_op;
    thread_reduce_<zero_init>(tensor, sum, sum_op);
}

template <typename T>
__device__ __forceinline__ T cuda_abs(T value);

// Specialization for __half (float16)
template <>
__device__ __forceinline__ __half cuda_abs<__half>(__half value) {
    return __habs(value);
}

// Specialization for __nv_bfloat16
template <>
__device__ __forceinline__ __nv_bfloat16 cuda_abs<__nv_bfloat16>(__nv_bfloat16 value) {
    return __habs(value);
}

// Specialization for float
template <>
__device__ __forceinline__ float cuda_abs<float>(float value) {
    return fabsf(value);
}

////////////////////////////////////////////////////////////////////////////////////////////////////


template <typename Engine0, typename Layout0>
__forceinline__ __device__ void subtract_row_max(Tensor<Engine0, Layout0> &scores, Tensor<Engine0, Layout0> &row_max) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(size<0>(scores) == size(row_max));
    #pragma unroll
    for (int mi = 0; mi < size<0>(scores); ++mi) {
        #pragma unroll
        for (int ni = 0; ni < size<1>(scores); ++ni) {
            scores(mi, ni) -= row_max(mi);
        }
    }
}

template <int Deg, typename Engine0, typename Layout0>
__forceinline__ __device__ void multiply_by_deg(Tensor<Engine0, Layout0> &scores) {
    if constexpr (Deg == 1) {
        return;
    } else if constexpr (Deg == 2) {
        CUTE_UNROLL
        for (int mi = 0; mi < size<0>(scores); ++mi) {
            CUTE_UNROLL
            for (int ni = 0; ni < size<1>(scores); ++ni) {
                scores(mi, ni) += scores(mi, ni); // deg2
            }
        }
    } else if constexpr (Deg == 3) {
        CUTE_UNROLL
        for (int mi = 0; mi < size<0>(scores); ++mi) {
            CUTE_UNROLL
            for (int ni = 0; ni < size<1>(scores); ++ni) {
                scores(mi, ni) += (scores(mi, ni) + scores(mi, ni)); // deg3
            }
        }
    } else if constexpr (Deg == 4) {
        CUTE_UNROLL
        for (int mi = 0; mi < size<0>(scores); ++mi) {
            CUTE_UNROLL
            for (int ni = 0; ni < size<1>(scores); ++ni) {
                scores(mi, ni) += scores(mi, ni); // deg2
                scores(mi, ni) += scores(mi, ni); // deg4
            }
        }
    } else if constexpr (Deg == 5) {
        CUTE_UNROLL
        for (int mi = 0; mi < size<0>(scores); ++mi) {
            CUTE_UNROLL
            for (int ni = 0; ni < size<1>(scores); ++ni) {
                auto deg1 = scores(mi, ni);
                scores(mi, ni) += scores(mi, ni); // deg2
                scores(mi, ni) += (deg1 + scores(mi, ni)); // deg5
            }
        }
    } else if constexpr (Deg == 6) {
        CUTE_UNROLL
        for (int mi = 0; mi < size<0>(scores); ++mi) {
            CUTE_UNROLL
            for (int ni = 0; ni < size<1>(scores); ++ni) {
                scores(mi, ni) += scores(mi, ni); // deg2
                scores(mi, ni) += (scores(mi, ni) + scores(mi, ni)); // deg6
            }
        }
    }
}

// Apply abslogp to all elements, subtract rowmax 
template <bool use_multiplier, int Deg, typename Engine0, typename Layout0>
__forceinline__ __device__ void apply_abslogp(Tensor<Engine0, Layout0> &scores, const float eps, const float log_stabilizer=0.0f) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");

    #pragma unroll
    for (int mi = 0; mi < size<0>(scores); ++mi) {
        #pragma unroll
        for (int ni = 0; ni < size<1>(scores); ++ni) {
            if (scores(mi, ni) != -INFINITY) {
                scores(mi, ni) = __logf(cuda_abs(scores(mi, ni)) + eps);
            }
        }
    }

    multiply_by_deg<Deg>(scores);

    if constexpr (use_multiplier) {
        #pragma unroll
        for (int i = 0; i < size(scores); ++i) {
            scores(i) -= log_stabilizer;
        }
    }
}

// Apply abs to all elements
template <typename Engine0, typename Layout0>
__forceinline__ __device__ auto apply_abs(Tensor<Engine0, Layout0> &scores) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    Tensor signs = make_tensor<bool>(scores.layout());
    #pragma unroll
    for (int mi = 0; mi < size<0>(scores); ++mi) {
        #pragma unroll
        for (int ni = 0; ni < size<1>(scores); ++ni) {
            signs(mi, ni) = scores(mi, ni) >= 0;
            scores(mi, ni) = cuda_abs(scores(mi, ni));
        }
    }
    return signs;
}

// Apply the exp to all the elements, without max
template <typename Engine0, typename Layout0>
__forceinline__ __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            #ifdef UNFUSE_FMA
                tensor(mi, ni) = exp2f(__fmul_rn(tensor(mi, ni), scale) - 0.0f);
            #else
                tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - 0.0f);
            #endif
        }
    }
}

// Apply exp2 to all the elements.
template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        // If we don't have float around M_LOG2E the multiplication is done in fp64.
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            // The following macro will disable the use of fma.
            // See: https://github.com/pytorch/pytorch/issues/121558 for more details
            // This macro is set in PyTorch and not FlashAttention
            #ifdef UNFUSE_FMA
                tensor(mi, ni) = exp2f(__fmul_rn(tensor(mi, ni), scale) - max_scaled);
            #else
                tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
            #endif
        }
    }
}

// Apply exp to all the elements.
template <bool use_fma=false, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        const float masked_max = max(mi) == -INFINITY ? 0.f : max(mi);
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            if constexpr (use_fma) {
                tensor(mi, ni) = expf(tensor(mi, ni) * 1.0f - masked_max);
            } else {
                tensor(mi, ni) = expf(tensor(mi, ni) - masked_max);
            }
        }
    }
}

template <int Deg>
__forceinline__ __device__ float _power(float x) {
    if constexpr (Deg == 1) {
        return x;
    } else if constexpr (Deg == 2) {
        return x * x;
    } else if constexpr (Deg == 3) {
        return x * x * x;
    } else if constexpr (Deg == 4) {
        return _power<2>(x) * _power<2>(x);
    }
}

// Apply power to all elements, divide by rowmax, multiply by signs
// This assumes tensor is already absed
template <int Deg, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Engine2, typename Layout2>
__forceinline__ __device__ void scale_apply_power(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, Tensor<Engine2, Layout2> const &signs) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor for tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor for max");
    static_assert(Layout2::rank == 2, "Only support 2D Tensor for signs");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(signs));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // if max is 0, then all elements must have been 0. (possibly due to masking).
        // We don't want 0 / 0 since that would give NaN.
        const float masked_max = max(mi) == 0.f ? 1.f : max(mi);
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni) {
            tensor(mi, ni) = _power<Deg>(tensor(mi, ni) / masked_max);
        }
    }
    if (Deg % 2 == 1) {
        #pragma unroll
        for (int mi = 0; mi < size<0>(tensor); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(tensor); ++ni) {
                tensor(mi, ni) *= signs(mi);
            }
        }
    }
}

// Apply power to all elements, divide by rowmax, multiply by signs
// This doesn't assume tensor is absed
template <int Deg, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_power(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor for tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor for max");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // if max is 0, then all elements must have been 0. (possibly due to masking).
        // We don't want 0 / 0 since that would give NaN.
        const float masked_max = max(mi) == 0.f ? 1.f : max(mi);
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni) {
            tensor(mi, ni) = _power<Deg>(tensor(mi, ni) / masked_max);
        }
    }
}


template <typename Tensor0, typename Tensor1>
__forceinline__ __device__ void apply_signs(Tensor0 &scores, Tensor1 &signs) {
    CUTE_STATIC_ASSERT_V(scores.layout() == signs.layout(), "score layout must be the same as signs layout");
    CUTE_UNROLL
    for (int i = 0; i < size(scores); i++) {
        scores(i) = signs(i) ? scores(i) : -scores(i);
    }
}

template <int kNRows, bool is_bf16, int Deg>
struct SymPower {

    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_max;

    __forceinline__ __device__ SymPower() = default;         

    /**
     * @brief Subtract rowmax and apply exp2 while rescaling output
     */
    template <bool Is_first, bool Check_inf=true, typename Tensor0, typename Tensor1, typename Tensor2>
    __forceinline__ __device__ void exp_with_rescale(Tensor0 &scores, Tensor1 &acc_o, Tensor2 &acc_norm) {
        static_assert(decltype(size(acc_norm))::value == kNRows);
        if (Is_first) {
            power::template reduce_max</*zero_init=*/true>(scores, row_max);
            power::template scale_apply_exp(scores, row_max);
            power::reduce_sum</*zero_init=*/true>(scores, acc_norm);
        } else {
            Tensor row_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, row_max_prev);
            power::template reduce_max</*zero_init=*/false>(scores, row_max);
            Tensor acc_o_rowcol = make_tensor(acc_o.data(), power::convert_layout_acc_rowcol(acc_o.layout()));
            static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
            #pragma unroll
            for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                float scores_ratio = expf((row_max_prev(mi) - scores_max_cur));
                acc_norm(mi) *= scores_ratio;
                #pragma unroll
                for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
                    acc_o_rowcol(mi, ni) *= scores_ratio;
                }
            }
            power::template scale_apply_exp(scores, row_max);
            power::reduce_sum</*zero_init=*/false>(scores, acc_norm);
        }
    }

    /**
     * @brief Subtract rowmax and apply exp2 while rescaling output, with signs
     */
    template <bool Is_first, bool Check_inf=true, typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3>
    __forceinline__ __device__ void exp_with_rescale(Tensor0 &scores, Tensor1 &acc_o, Tensor2 &acc_norm, Tensor3 &signs) {
        static_assert(decltype(size(acc_norm))::value == kNRows);
        if (Is_first) {
            power::template reduce_max</*zero_init=*/true>(scores, row_max);
            power::template scale_apply_exp(scores, row_max);
            power::template apply_signs(scores, signs);
            power::reduce_sum</*zero_init=*/true>(scores, acc_norm);
        } else {
            Tensor row_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, row_max_prev);
            power::template reduce_max</*zero_init=*/false>(scores, row_max);
            Tensor acc_o_rowcol = make_tensor(acc_o.data(), power::convert_layout_acc_rowcol(acc_o.layout()));
            static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
            #pragma unroll
            for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                float scores_ratio = expf((row_max_prev(mi) - scores_max_cur));
                acc_norm(mi) *= scores_ratio;
                #pragma unroll
                for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
                    acc_o_rowcol(mi, ni) *= scores_ratio;
                }
            }
            power::template scale_apply_exp(scores, row_max);
            power::template apply_signs(scores, signs);
            power::reduce_sum</*zero_init=*/false>(scores, acc_norm);
        }
    }

    /**
     * @brief Subtract rowmax and apply exp2 while rescaling output
     */
    template <bool Is_first, bool Check_inf=true, typename Tensor0, typename Tensor1, typename Tensor2>
    __forceinline__ __device__ void power_with_rescale(Tensor0 &scores, Tensor1 &acc_o, Tensor2 &acc_norm) {
        static_assert(decltype(size(acc_norm))::value == kNRows);
        if (Is_first) {
            power::template reduce_absmax</*zero_init=*/true>(scores, row_max);
            power::template scale_apply_power<Deg>(scores, row_max);
            power::reduce_sum</*zero_init=*/true>(scores, acc_norm);
        } else {
            Tensor row_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, row_max_prev);
            power::template reduce_absmax</*zero_init=*/false>(scores, row_max);
            Tensor acc_o_rowcol = make_tensor(acc_o.data(), power::convert_layout_acc_rowcol(acc_o.layout()));
            static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
            #pragma unroll
            for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == 0.0f ? 1.0f : row_max(mi));
                float scores_ratio = _power<Deg>(row_max_prev(mi) / scores_max_cur);
                acc_norm(mi) *= scores_ratio;
                #pragma unroll
                for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
                    acc_o_rowcol(mi, ni) *= scores_ratio;
                }
            }
            power::template scale_apply_power<Deg>(scores, row_max);
            power::reduce_sum</*zero_init=*/false>(scores, acc_norm);
        }
    }
};

}  // namespace power
