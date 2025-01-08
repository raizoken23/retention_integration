#pragma once

#include <assert.h>
#include <type_traits>

#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>


#ifndef FETCH_INT4
#define FETCH_INT4(pointer) (reinterpret_cast<int4 *>(&(pointer))[0])
#endif
#ifndef FETCH_INT2
#define FETCH_INT2(pointer) (reinterpret_cast<int2 *>(&(pointer))[0])
#endif
#ifndef FETCH_VECINT
#define FETCH_VECINT(pointer, size) ((size) == 4 ? FETCH_INT4(pointer) : FETCH_INT2(pointer))
#endif


template<typename T, typename... Types>
struct is_one_of : std::false_type {};

template<typename T, typename First, typename... Rest>
struct is_one_of<T, First, Rest...> 
    : std::conditional_t<std::is_same<T, First>::value, std::true_type, is_one_of<T, Rest...>> {};


namespace state_kernel
{
    using namespace cute;

    // same as cute::cp_async_wait except for no fence for N = 0
    template <int N>
    CUTE_HOST_DEVICE void cp_async_wait()
    {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
        asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
    }

    // Borrowed from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/utils.h
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    struct MaxOp
    {
        __device__ __forceinline__ T operator()(T const &x, T const &y) { return x > y ? x : y; }
    };

    template <>
    struct MaxOp<float>
    {
        // This is slightly faster
        __device__ __forceinline__ float operator()(float const &x, float const &y) { return max(x, y); }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    struct SumOp
    {
        __device__ __forceinline__ T operator()(T const &x, T const &y) { return x + y; }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <int THREADS>
    struct Allreduce
    {
        static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
        template <typename T, typename Operator>
        static __device__ __forceinline__ T run(T x, Operator &op)
        {
            constexpr int OFFSET = THREADS / 2;
            x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
            return Allreduce<OFFSET>::run(x, op);
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    struct Allreduce<2>
    {
        template <typename T, typename Operator>
        static __device__ __forceinline__ T run(T x, Operator &op)
        {
            x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
            return x;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    
    template <int MOD>
    struct AllreduceMod
    {
        static_assert(MOD == 1 || MOD == 2 || MOD == 4 || MOD == 8);
        template <typename T, typename Operator>
        static __device__ __forceinline__ T run(T x, Operator &op)
        {
            x = op(x, __shfl_xor_sync(uint32_t(-1), x, MOD));
            constexpr int OFFSET = MOD * 2;
            return AllreduceMod<OFFSET>::run(x, op);
        }
    };

    template <>
    struct AllreduceMod<16>
    {
        template <typename T, typename Operator>
        static __device__ __forceinline__ T run(T x, Operator &op)
        {
            x = op(x, __shfl_xor_sync(uint32_t(-1), x, 16));
            return x;
        }
    };


    // Convert acc_layout from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
    template <typename Layout>
    __forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout)
    {
        static_assert(decltype(size<0>(acc_layout))::value == 4);
        static_assert(decltype(rank(acc_layout))::value == 3);
        auto l = logical_divide(acc_layout, Shape<_2>{}); // ((2, 2), MMA_M, MMA_N)
        return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
    };

    // Convert rA_layout from (MMA=(2, 2, 2), MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, 2, MMA_K))
    template <typename Layout>
    __forceinline__ __device__ auto convert_layout_rA_rowcol(Layout regA_layout)
    {
        static_assert(decltype(size<0>(regA_layout))::value == 8);
        static_assert(decltype(rank(get<0>(regA_layout)))::value == 3);
        static_assert(decltype(size<0, 0>(regA_layout))::value == 2);
        static_assert(decltype(size<0, 1>(regA_layout))::value == 2);
        static_assert(decltype(size<0, 2>(regA_layout))::value == 2);
        static_assert(decltype(rank(regA_layout))::value == 3);
        auto l = regA_layout; // ((2, 2, 2)), MMA_M, MMA_N)
        return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), make_layout(get<0, 2>(l), get<2>(l))));
    };

    // // Convert rB_layout from (MMA=(2, 2), MMA_N, MMA_K) to (nrow=(2, MMA_N), ncol=(2, MMA_K))
    // template <typename Layout>
    // __forceinline__ __device__ auto convert_layout_rB_rowcol(Layout regB_layout) {
    //     static_assert(decltype(size<0>(regB_layout))::value == 4);
    //     static_assert(decltype(rank(get<0>(regB_layout)))::value == 3);
    //     static_assert(decltype(size<0, 0>(regB_layout))::value == 2);
    //     static_assert(decltype(size<0, 1>(regB_layout))::value == 2);
    //     return make_layout(make_layout(get<0, 1>(regB_layout), get<1>(regB_layout)), make_layout(get<0, 0>(regB_layout), get<2>(regB_layout)));
    // }

    template <bool zero_init = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
    __device__ __forceinline__ void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op)
    {
        static_assert(Layout0::rank == 2, "Only support 2D Tensor");
        static_assert(Layout1::rank == 1, "Only support 1D Tensor");
        CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
#pragma unroll
        for (int mi = 0; mi < size<0>(tensor); mi++)
        {
            summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
#pragma unroll
            for (int ni = 1; ni < size<1>(tensor); ni++)
            {
                summary(mi) = op(summary(mi), tensor(mi, ni));
            }
        }
    };

    template <bool zero_init = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
    __device__ __forceinline__ void thread_reduce_1d(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op, int selector)
    {
        // static_assert(Layout0::rank == 1, "Only support 1D Tensor");
        // CUTE_STATIC_ASSERT_V(size<0>(summary) == 1, "Only support reducing to single number");
        summary(0, selector) = zero_init ? tensor(0) : op(summary(0, selector), tensor(0));
#pragma unroll
        for (int mi = 1; mi < size<0>(tensor); mi++)
        {
            summary(0, selector) = op(summary(0, selector), tensor(mi));
        }
    }

    template <typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
    __device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op)
    {
        CUTE_STATIC_ASSERT_V(size(dst) == size(src));
#pragma unroll
        for (int i = 0; i < size(dst); i++)
        {
            dst(i) = Allreduce<4>::run(src(i), op);
        }
    };

    template <typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
    __device__ __forceinline__ void quad_allreduce_mod_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op)
    {
        CUTE_STATIC_ASSERT_V(size(dst) == size(src));
#pragma unroll
        for (int i = 0; i < size(dst); i++)
        {
            dst(i) = AllreduceMod<4>::run(src(i), op);
        }
    };

    template <bool zero_init = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
    __device__ __forceinline__ void reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op)
    {
        thread_reduce_<zero_init>(tensor, summary, op);
        quad_allreduce_(summary, summary, op);
    };

    template <bool zero_init = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
    __device__ __forceinline__ void reduce_max(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &max)
    {
        MaxOp<float> max_op;
        reduce_<zero_init>(tensor, max, max_op);
    };

    template <bool zero_init = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
    __device__ __forceinline__ void reduce_sum_thread(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &sum)
    {
        SumOp<float> sum_op;
        thread_reduce_<zero_init>(tensor, sum, sum_op);
    };

    template <bool zero_init = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
    __device__ __forceinline__ void reduce_sum(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &sum)
    {
        SumOp<float> sum_op;
        reduce_<zero_init>(tensor, sum, sum_op);
    };

    template <typename Engine0, typename Layout, typename Engine1>
    __forceinline__ __device__ auto convert_type(Tensor<Engine0, Layout> const &tensor, Tensor<Engine1, Layout> &output)
    {
        using From_type = typename Engine0::value_type;
        using To_type = typename Engine1::value_type;
        constexpr int numel = decltype(size(tensor))::value;
        cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
        auto from_array = reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data());
        auto to_array = reinterpret_cast<cutlass::Array<To_type, numel> *>(output.data());

        *to_array = convert_op(*from_array);
    };

    template <typename To_type, typename Engine, typename Layout>
    __forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
        using From_type = typename Engine::value_type;
        constexpr int numel = decltype(size(tensor))::value;
        cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
        // HACK: this requires tensor to be "contiguous"
        auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
        return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
    };

    // copy 2d tensor from source to dest
    template <typename TiledCopy, typename Engine0, typename Layout0,
              typename Engine1, typename Layout1>
    __forceinline__ __device__ void copy2d(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S, Tensor<Engine1, Layout1> const &D)
    {
        CUTE_UNROLL
        for (int i = 0; i < size<1>(S); ++i)
        {
            CUTE_UNROLL
            for (int j = 0; j < size<2>(S); ++j)
            {
                cute::copy(tiled_copy, S(_, i, j), D(_, i, j));
            }
        }
    };

    template <bool Is_even_MN=true, bool Clear_OOB_MN=false, int block_size=0,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2>
    __forceinline__ __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN, const int max_MN=0) {
        static_assert(decltype(rank(S))::value == 3 || decltype(rank(S))::value == 2);
        if constexpr (decltype(rank(S))::value == 3) {
            CUTE_STATIC_ASSERT_V(rank(D) == _3{});
            CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
            CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
            CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K

            #pragma unroll
            for (int m = 0; m < size<1>(S); ++m) {
                if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
                    #pragma unroll
                    for (int k = 0; k < size<2>(S); ++k) {
                        cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
                    }
                } else if (Clear_OOB_MN && (block_size == 0 || get<0>(identity_MN(0, m, 0)) < block_size)) {
                    cute::clear(D(_, m, _));
                }
            }
        } else if constexpr (decltype(rank(S))::value == 2) {
            CUTE_STATIC_ASSERT_V(rank(D) == _2{});
            CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
            CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M

            #pragma unroll
            for (int m = 0; m < size<1>(S); ++m) {
                if (Is_even_MN || get<0>(identity_MN(0, m)) < max_MN) {
                    cute::copy(tiled_copy, S(_, m), D(_, m));
                } else if (Clear_OOB_MN && (block_size == 0 || get<0>(identity_MN(0, m)) < block_size)) {
                    cute::clear(D(_, m));
                }
            }
        }
    };

    template <bool Is_even_MN=true, bool Clear_OOB_MN=false, bool Is_even_K=true, bool Clear_OOB_K=true,
            typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
            typename Engine2, typename Layout2, typename Engine3, typename Layout3>
    __forceinline__ __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN, Tensor<Engine3, Layout3> const &predicate_K, const int max_MN=0) {
        CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
        CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
        CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
        CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K

        #pragma unroll
        for (int m = 0; m < size<1>(S); ++m) {
            if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
                #pragma unroll
                for (int k = 0; k < size<2>(S); ++k) {
                    if (Is_even_K || predicate_K(k)) {
                        cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
                    } else if (Clear_OOB_K) {
                        cute::clear(D(_, m, k));
                    }
                }
            } else if (Clear_OOB_MN) {
                cute::clear(D(_, m, _));
            }
        }
    };

    template <typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
    __forceinline__ __device__ void safe_copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S, Tensor<Engine1, Layout1> const &D, const int total_copy_size, const int NThreads, const int tid) {
        if (total_copy_size >= NThreads) {
            cute::copy(tiled_copy, S, D);
        }
        else if (tid < total_copy_size) {
            cute::copy(tiled_copy, S, D);
        }
    }

    // copy 2d tensor from source to dest
    template <typename TiledCopy, typename Engine0, typename Layout0,
              typename Engine1, typename Layout1>
    __forceinline__ __device__ void copy1d(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S, Tensor<Engine1, Layout1> const &D)
    {
        CUTE_UNROLL
        for (int i = 0; i < size<1>(S); ++i)
        {
            cute::copy(tiled_copy, S(_, i), D(_, i));
        }
    };

    template <typename TensorA, typename TensorB, typename TensorC>
    __forceinline__ __device__ void add_(const TensorA&& a, const TensorB&& b, TensorC&& c) {
        CUTE_STATIC_ASSERT_V(size(a) == size(b), "Tensor sizes must match");
        CUTE_STATIC_ASSERT_V(size(a) == size(c), "Tensor sizes must match");
        CUTE_UNROLL
        for (int i = 0; i < size(a); i++) {
            c[i] = a[i] + b[i];
        }
    }

    template <typename TensorA, typename TensorB, typename TensorC>
    __forceinline__ __device__ void multiply_(const TensorA& a, const TensorB& b, TensorC& c) {
        CUTE_STATIC_ASSERT_V(size(a) == size(b), "Tensor sizes must match");
        CUTE_STATIC_ASSERT_V(size(a) == size(c), "Tensor sizes must match");
        CUTE_UNROLL
        for (int i = 0; i < size(a); i++) {
            c[i] = a[i] * b[i];
        }
    }

    template <typename TensorA, typename TensorB>
    __forceinline__ __device__ void assign_(TensorA a, const TensorB& b) {
        CUTE_STATIC_ASSERT_V(size(a) == size(b), "Tensor sizes must match ");
        CUTE_UNROLL
        for (int i = 0; i < size(a); i++) {
            a[i] = b[i];
        }
    }

    template <typename TensorA, typename TensorB, typename TensorC>
    __forceinline__ __device__ void apply_sign_(const TensorA&& a, const TensorB&& signs, TensorC&& c) {
        CUTE_STATIC_ASSERT_V(size(a) == size(c), "Tensor sizes must match");
        CUTE_STATIC_ASSERT_V(size(a) == size(signs), "Tensor sizes must match");
        static_assert(std::is_same_v<typename TensorB::element_type, bool>, "TensorB must be bool");
        CUTE_UNROLL
        for (int i = 0; i < size(a); i++) {
            c[i] = signs[i] ? a[i] : -a[i];
        }
    }
    /**
     * @brief Applies a unary function to each element of a tensor, optionally converting to FP32 for computation.
     * 
     * @tparam TensorA The input tensor type.
     * @tparam Func The unary function type to apply.
     * @tparam reuse_input A flag to try to reuse the input tensor for the result if possible (default: true).
     * 
     * @param a The input tensor.
     * @param func The unary function to apply to each element.
     * 
     * @return A tensor with same layout and data type as the input tensor, but with each element transformed by the unary function.
     * 
     * @note For cutlass::tfloat32_t and float, it uses the original precision unless fp32 is explicitly set to true.
     *       For other types, it always converts to float for computation.
     */
    template <typename TensorA, typename Func, bool reuse_input = true, bool return_fp32 = false>
    __forceinline__ __device__ auto fp32_op_(TensorA&& a, Func&& func) {
        using Element = typename TensorA::element_type;
        using float_type = float;
        using TensorA_fp32 = decltype(make_tensor<float_type>(a.layout()));
        TensorA_fp32 a_fp32;
        if constexpr (reuse_input && is_one_of<Element, cutlass::tfloat32_t, float>::value) {
            a_fp32 = a;
        } else {
            a_fp32 = state_kernel::convert_type<float_type>(a);
        }
        CUTE_UNROLL
        for (int i = 0; i < size(a_fp32); i++) {
            a_fp32[i] = func(a_fp32[i]);
        }
        if constexpr (return_fp32) {
            return a_fp32;
        } else {
            if constexpr (reuse_input && !is_one_of<Element, cutlass::tfloat32_t, float>::value) {
                state_kernel::convert_type(a_fp32, a);
                return a;
            } else {
                return state_kernel::convert_type<Element>(a_fp32);
            }
        }
    }

    template <typename TensorA, bool reuse_input = true, bool return_fp32 = false>
    __forceinline__ __device__ auto logf_(TensorA&& a) {
        return fp32_op_<TensorA, decltype(logf), reuse_input, return_fp32>(a, logf);
    }

    template <typename TensorA, bool reuse_input = true, bool return_fp32 = false>
    __forceinline__ __device__ auto expf_(TensorA&& a) {
        return fp32_op_<TensorA, decltype(expf), reuse_input, return_fp32>(a, expf);
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

    template <bool reuse_input = true, bool return_fp32 = false, typename TensorA, 
              typename Element = typename TensorA::element_type,
              typename = std::enable_if_t<std::is_same_v<Element, cutlass::half_t> || std::is_same_v<Element, cutlass::bfloat16_t>>>
    __forceinline__ __device__ auto logabsf_(TensorA&& a) {
        using float_type = float;
        using TensorA_fp32 = decltype(make_tensor<float_type>(a.shape()));
        using TensorA_sign = decltype(make_tensor<bool>(a.shape()));
        TensorA_fp32 a_fp32 = make_tensor<float_type>(a.shape());
        TensorA_sign signs = make_tensor<bool>(a.shape()); // TODO(sean): can we use one bit to store the sign?

        CUTE_UNROLL
        for (int i = 0; i < size(a); i++) {
            signs[i] = a[i] >= 0;
            a_fp32[i] = logf(static_cast<float_type>(cuda_abs(a[i])));
        }

        if constexpr (return_fp32) {
            return std::make_pair(signs, a_fp32);
        } else {
            if constexpr (reuse_input) {
                state_kernel::convert_type(a_fp32, a);
                return std::make_pair(signs, a);
            } else {
                return std::make_pair(signs, state_kernel::convert_type<Element>(a_fp32));
            }
        }
    }

    template <typename TensorA, typename TensorB, typename TensorC>
    __forceinline__ __device__ void xor_(const TensorA&& a, const TensorB&& b, TensorC&& c) {
        CUTE_STATIC_ASSERT_V(size(a) == size(b), "Tensor sizes must match");
        CUTE_STATIC_ASSERT_V(size(a) == size(c), "Tensor sizes must match");
        static_assert(std::is_same_v<typename TensorA::element_type, bool>, "TensorA must be bool");
        static_assert(std::is_same_v<typename TensorB::element_type, bool>, "TensorB must be bool");
        static_assert(std::is_same_v<typename TensorC::element_type, bool>, "TensorC must be bool");
        CUTE_UNROLL
        for (int i = 0; i < size(a); i++) {
            c[i] = a[i] != b[i];
        }
    }

    template <typename TensorA, typename Multiplier_t, typename TensorC>
    __forceinline__ __device__ void elementwise_product(TensorA &&A, Multiplier_t B, TensorC &&C)
    {
        CUTE_STATIC_ASSERT_V(size(A) == size(C), "Tensor sizes must match");
        using Element_1 = typename std::decay_t<TensorA>::value_type;
        using Element_2 = typename std::decay_t<TensorC>::value_type;
        CUTE_UNROLL
        for (int i = 0; i < size(A); ++i)
        {
            C(i) = static_cast<Element_2>(A(i) * B);
        }
    };

    template <bool fused=false,typename T1, typename T2>
    __forceinline__ __device__ auto fp32_mul(T1 a, T2 b) {
        if constexpr (fused) {
            return static_cast<float>(a) * static_cast<float>(b);
        } else {
            return static_cast<float>(a) * static_cast<float>(b) + 0.0f;
        }
    }

    template <bool A_in_regs = false, bool B_in_regs = false, bool rescale_B = false, bool rescale_A = false, typename Tensor0, typename Tensor1,
              typename Tensor2, typename Tensor3, typename Tensor4,
              typename TiledMma, typename TiledCopyA, typename TiledCopyB,
              typename ThrCopyA, typename ThrCopyB>
    __forceinline__ __device__ void gemm(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const &tCsA,
                                         Tensor4 const &tCsB, TiledMma tiled_mma,
                                         TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                                         ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B, float multiplierB = 1.0f, float multiplierA = 1.0f)
    {
        CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));  // MMA_M
        CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));  // MMA_N
        CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB)); // MMA_K
        Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
        CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view)); // M
        Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);

        CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view)); // N
        if constexpr (!A_in_regs)
        {
            cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
        }
        if constexpr (!B_in_regs)
        {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
        }
#pragma unroll
        for (int i = 0; i < size<2>(tCrA); ++i) // MMA_K
        {
            if (i < size<2>(tCrA) - 1)
            {
                if constexpr (!A_in_regs)
                {
                    cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
                }
                if constexpr (!B_in_regs)
                {
                    cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
                }
            }
            if constexpr (rescale_A)
            {
                Tensor srA = tCrA(_, _, i);
                elementwise_product(srA, multiplierA, srA);
            }
            if constexpr (rescale_B)
            {
                Tensor srB = tCrB(_, _, i);
                elementwise_product(srB, multiplierB, srB);
            }
            cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <bool A_in_regs = false, bool B_in_regs = false, bool rescale_B = false, bool rescale_A = false, typename Tensor0, typename Tensor1, typename Tensor2,
              typename Tensor3, typename Tensor4, typename Tensor5,
              typename TiledMma, typename TiledCopyA, typename TiledCopyB,
              typename ThrCopyA, typename ThrCopyB>
    __forceinline__ __device__ void gemm_an(Tensor0 &acc, Tensor1 &acc_an, Tensor2 &tCrA, Tensor3 &tCrB, Tensor4 const &tCsA,
                                         Tensor5 const &tCsB, TiledMma tiled_mma,
                                         TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                                         ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B, float multiplierB = 1.0f, float multiplierA = 1.0f)
    {
        CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));  // MMA_M
        CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));  // MMA_N
        CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB)); // MMA_K
        Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
        CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view)); // M
        Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
        Tensor tCrA_rowcol = make_tensor(tCrA.data(), convert_layout_rA_rowcol(tCrA.layout()));
        using acc_an_type = typename Tensor0::value_type;

        CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view)); // N
        if (!A_in_regs)
        {
            cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
        }
        if (!B_in_regs)
        {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
        }

        #pragma unroll
        for (int i = 0; i < size<2>(tCrA); ++i) // MMA_K
        {
            if (i < size<2>(tCrA) - 1)
            {
                if (!A_in_regs)
                {
                    cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
                }
                if (!B_in_regs)
                {
                    cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
                }
            }
            if (rescale_A)
            {
                Tensor srA = tCrA(_, _, i);
                elementwise_product(srA, multiplierA, srA);
            }
            if (rescale_B)
            {
                Tensor srB = tCrB(_, _, i);
                elementwise_product(srB, multiplierB, srB);
            }
            cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);

            // accumulate norm
            #pragma unroll
            for (int j = 0; j < size<0>(tCrA_rowcol); ++j) {
                #pragma unroll
                for (int l = 0; l < size<1, 0>(tCrA_rowcol); ++l) {
                    #pragma unroll
                    for (int m = 0; m < size<1, 1>(tCrA_rowcol); ++m) {
                        acc_an[j] += static_cast<acc_an_type>(tCrA_rowcol(j, make_coord(l, make_coord(m, i))));
                    }
                }
            }
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template<bool rescale_B = false, typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
            typename TiledMma, typename TiledCopy, typename ThrCopy>
    __forceinline__ __device__ void gemm_rs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                                TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                                ThrCopy smem_thr_copy_B, float multiplierB = 1.0f) {
        CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
        CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
        CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
        Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
        CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
        cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
        #pragma unroll
        for (int i = 0; i < size<2>(tCrA); ++i) {
            if (i < size<2>(tCrA) - 1)
            {
                cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
            }
            if constexpr (rescale_B)
            {
                elementwise_product(tCrB(_, _, i), multiplierB, tCrB(_, _, i));
            }
            if (i < size<2>(tCrA) - 1) {
                cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
            }
            cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
        }
    }

    // template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
    //         typename TiledMma, typename TiledCopy, typename ThrCopy, typename Tensor4>
    // __forceinline__ __device__ void gemm_rs_scale_K_B(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
    //                             TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
    //                             ThrCopy smem_thr_copy_B, Tensor4 &multiplierB) {
    //     CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    //     CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    //     CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    //     Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    //     CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    //     Tensor tCrB_rowcol = make_tensor(tCrB.data(), convert_layout_rB_rowcol(tCrB.layout()));
    //     CUTE_STATIC_ASSERT_V(size(multiplierB) == size<1>(tCrB_rowcol));
    //     cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    //     #pragma unroll
    //     for (int i = 0; i < size<2>(tCrA); ++i) {
    //         if (i < size<2>(tCrA) - 1)
    //         {
    //             cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
    //         }

    //         elementwise_product(tCrB(_, _, i), multiplierB, tCrB(_, _, i));
    //         if (i < size<2>(tCrA) - 1) {
    //             cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
    //         }
    //         cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    //     }
    // }



    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template<bool rescale_B = false, typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3, typename Tensor4,
            typename TiledMma, typename TiledCopy, typename ThrCopy>
    __forceinline__ __device__ void gemm_rs_an(Tensor0 &acc, Tensor1 &acc_an, Tensor2 &tCrA, Tensor3 &tCrB, Tensor4 const& tCsB,
                                TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                                ThrCopy smem_thr_copy_B, float multiplierB = 1.0f) {
        using acc_an_type = typename Tensor0::value_type;
        CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
        CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
        CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
        Tensor tCrA_rowcol = make_tensor(tCrA.data(), convert_layout_rA_rowcol(tCrA.layout()));
        Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
        CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
        cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
        if constexpr (rescale_B) {
            Tensor srB = tCrB(_, _, 0);
            elementwise_product(srB, multiplierB, srB);
        }
        #pragma unroll
        for (int i = 0; i < size<2>(tCrA); ++i) {
            if (i < size<2>(tCrA) - 1) {
                cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
            }
            // accumulate norm
            #pragma unroll
            for (int j = 0; j < size<0>(tCrA_rowcol); ++j) {
                #pragma unroll
                for (int l = 0; l < size<1, 0>(tCrA_rowcol); ++l) {
                    #pragma unroll
                    for (int m = 0; m < size<1, 1>(tCrA_rowcol); ++m) {
                        acc_an[j] += static_cast<acc_an_type>(tCrA_rowcol(j, make_coord(l, make_coord(m, i))));
                    }
                }
            }
            cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
        }  
    }

} // namespace state_kernel