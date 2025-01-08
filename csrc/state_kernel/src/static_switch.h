// Inspired by https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h

#pragma once

#include <torch/python.h>
#include "kernel_traits.h"

#define BOOL_SWITCH(COND, CONST_NAME, ...)            \
    [&] {                                             \
        if (COND)                                     \
        {                                             \
            constexpr static bool CONST_NAME = true;  \
            return __VA_ARGS__();                     \
        }                                             \
        else                                          \
        {                                             \
            constexpr static bool CONST_NAME = false; \
            return __VA_ARGS__();                     \
        }                                             \
    }()

#ifdef FAST_IS_EVEN_MN
    #define EVEN_MN_SWITCH(COND, CONST_NAME, ...) \
        [&] { \
            constexpr static bool CONST_NAME = FAST_IS_EVEN_MN; \
            return __VA_ARGS__(); \
        }()
#else
    #define EVEN_MN_SWITCH BOOL_SWITCH
#endif

#define INT_SWITCH(VALUE, CONST_NAME, ...)       \
    [&] {                                        \
        constexpr static int CONST_NAME = VALUE; \
        return __VA_ARGS__();                    \
    }()

#ifdef FAST_HEAD_DIM
    #define STATE_HEADDIM_SWITCH(HEADDIM, CONST_NAME, ...) \
        [&] { \
            constexpr static int CONST_NAME = FAST_HEAD_DIM; \
            return __VA_ARGS__(); \
        }()
#else
    #define STATE_HEADDIM_SWITCH(HEADDIM, CONST_NAME, ...)        \
        [&] {                                                     \
            if (HEADDIM <= 32)                                    \
            {                                                     \
                constexpr static int CONST_NAME = 32;             \
                return __VA_ARGS__();                             \
            }                                                     \
            else if (HEADDIM <= 64)                               \
            {                                                     \
                constexpr static int CONST_NAME = 64;             \
                return __VA_ARGS__();                             \
            }                                                     \
            else                                                  \
            {                                                     \
                TORCH_CHECK(false, "Unsupported head dimension"); \
                return;                                           \
            }                                                     \
        }()
#endif

#ifdef FAST_STATE_DEG
    #define STATE_DEG_SWITCH(DEG, CONST_NAME, ...) \
        [&] { \
            constexpr static int CONST_NAME = FAST_STATE_DEG; \
            return __VA_ARGS__(); \
        }()
#else
    #define STATE_DEG_SWITCH(DEG, CONST_NAME, ...)                      \
        [&] {                                                           \
            if (DEG == 2)                                               \
            {                                                           \
                constexpr static int CONST_NAME = 2;                    \
                return __VA_ARGS__();                                   \
            }                                                           \
            else if (DEG == 4)                                          \
            {                                                           \
                constexpr static int CONST_NAME = 4;                    \
                return __VA_ARGS__();                                   \
            }                                                           \
            else                                                        \
            {                                                           \
                TORCH_CHECK(false, "Unsupported degree of similarity"); \
                return;                                                 \
            }                                                           \
        }()
#endif

#ifdef FAST_IS_FP16
    #if FAST_IS_FP16
        #define STATE_DTYPE_SWITCH(COND, CONST_NAME, ...) \
            [&] { \
                using CONST_NAME = cutlass::half_t; \
                return __VA_ARGS__(); \
            }()
    #else
        #define STATE_DTYPE_SWITCH(COND, CONST_NAME, ...) \
            [&] { \
                using CONST_NAME = cutlass::bfloat16_t; \
                return __VA_ARGS__(); \
            }()
    #endif
#else
    #define STATE_DTYPE_SWITCH(COND, CONST_NAME, ...)   \
        [&] {                                           \
            if (COND)                                   \
            {                                           \
                using CONST_NAME = cutlass::half_t;     \
                return __VA_ARGS__();                   \
            }                                           \
            else                                        \
            {                                           \
                using CONST_NAME = cutlass::bfloat16_t; \
                return __VA_ARGS__();                   \
            }                                           \
        }()
#endif

#define DTYPE_SWITCH(DTYPE, CONST_NAME, ...)         \
    [&] {                                            \
        if (DTYPE == torch::kFloat32)                \
        {                                            \
            using CONST_NAME = float;                \
            return __VA_ARGS__();                    \
        }                                            \
        else if (DTYPE == torch::kFloat16)           \
        {                                            \
            using CONST_NAME = cutlass::half_t;      \
            return __VA_ARGS__();                    \
        }                                            \
        else if (DTYPE == torch::kBFloat16)          \
        {                                            \
            using CONST_NAME = cutlass::bfloat16_t;  \
            return __VA_ARGS__();                    \
        }                                            \
        else                                         \
        {                                            \
            TORCH_CHECK(false, "Unsupported dtype"); \
            return;                                  \
        }                                            \
    }()


#define BINARY_DIM_SWITCH_2(VALUE, CONST_NAME, LAMBDA) \
    [&] {                                              \
        if (VALUE == 0)                                \
        {                                              \
            constexpr static int CONST_NAME = 0;       \
            return LAMBDA();                           \
        }                                              \
        else                                           \
        {                                              \
            constexpr static int CONST_NAME = 1;       \
            return LAMBDA();                           \
        }                                              \
    }()

#define BINARY_DIM_SWITCH_4(VALUE, CONST_NAME, LAMBDA)                             \
    [&] {                                                                          \
        if (VALUE < 2)                                                             \
        {                                                                          \
            return BINARY_DIM_SWITCH_2(VALUE, CONST_NAME, LAMBDA);                 \
        }                                                                          \
        else                                                                       \
        {                                                                          \
            return BINARY_DIM_SWITCH_2(VALUE - 2, CONST_NAME##_offset, [&]() { \
            constexpr static int CONST_NAME = CONST_NAME##_offset + 2;  \
            return LAMBDA(); }); \
        }                                                                          \
    }()

#define BINARY_DIM_SWITCH_8(VALUE, CONST_NAME, LAMBDA)                             \
    [&] {                                                                          \
        if (VALUE < 4)                                                             \
        {                                                                          \
            return BINARY_DIM_SWITCH_4(VALUE, CONST_NAME, LAMBDA);                 \
        }                                                                          \
        else                                                                       \
        {                                                                          \
            return BINARY_DIM_SWITCH_4(VALUE - 4, CONST_NAME##_offset, [&]() { \
            constexpr static int CONST_NAME = CONST_NAME##_offset + 4;  \
            return LAMBDA(); }); \
        }                                                                          \
    }()

#define BINARY_DIM_SWITCH_16(VALUE, CONST_NAME, LAMBDA)                     \
    [&] {                                                                   \
        if (VALUE < 8)                                                      \
        {                                                                   \
            BINARY_DIM_SWITCH_8(VALUE, CONST_NAME, LAMBDA);                 \
        }                                                                   \
        else                                                                \
        {                                                                   \
            BINARY_DIM_SWITCH_8(VALUE - 8, CONST_NAME##_offset, [&]() {     \
            constexpr static int CONST_NAME = CONST_NAME##_offset + 8;  \
                return LAMBDA(); }); \
        }                                                                   \
    }()

#define BINARY_DIM_SWITCH_32(VALUE, CONST_NAME, LAMBDA)                              \
    [&] {                                                                            \
        if (VALUE < 16)                                                              \
        {                                                                            \
            return BINARY_DIM_SWITCH_16(VALUE, CONST_NAME, LAMBDA);                  \
        }                                                                            \
        else                                                                         \
        {                                                                            \
            return BINARY_DIM_SWITCH_16(VALUE - 16, CONST_NAME##_offset, [&]() { \
                    constexpr static int CONST_NAME = CONST_NAME##_offset + 16;   \
                    return LAMBDA(); }); \
        }                                                                            \
    }()

#define BINARY_DIM_SWITCH_64(VALUE, CONST_NAME, LAMBDA)                              \
    [&] {                                                                            \
        if (VALUE < 32)                                                              \
        {                                                                            \
            return BINARY_DIM_SWITCH_32(VALUE, CONST_NAME, LAMBDA);                  \
        }                                                                            \
        else                                                                         \
        {                                                                            \
            return BINARY_DIM_SWITCH_32(VALUE - 32, CONST_NAME##_offset, [&]() { \
                constexpr static int CONST_NAME = CONST_NAME##_offset + 32;   \
                return LAMBDA(); }); \
        }                                                                            \
    }()

#define BINARY_DIM_SWITCH_128(VALUE, CONST_NAME, LAMBDA)                             \
    [&] {                                                                            \
        if (VALUE < 64)                                                              \
        {                                                                            \
            return BINARY_DIM_SWITCH_64(VALUE, CONST_NAME, LAMBDA);                  \
        }                                                                            \
        else                                                                         \
        {                                                                            \
            return BINARY_DIM_SWITCH_64(VALUE - 64, CONST_NAME##_offset, [&]() {  \
                constexpr static int CONST_NAME = CONST_NAME##_offset + 64;       \
                return LAMBDA(); }); \
        }                                                                            \
    }()

#define BINARY_DIM_SWITCH(VALUE, CONST_NAME, HEADDIM, LAMBDA)                                                                                                              \
    [&] {                                                                                                                                                                  \
        if constexpr (HEADDIM == 1)                                                                                                                                        \
        {                                                                                                                                                                  \
            constexpr static int CONST_NAME = 0;                                                                                                                           \
            return LAMBDA();                                                                                                                                               \
        }                                                                                                                                                                  \
        else if constexpr (HEADDIM == 2)                                                                                                                                   \
        {                                                                                                                                                                  \
            return BINARY_DIM_SWITCH_2(VALUE, CONST_NAME, LAMBDA);                                                                                                         \
        }                                                                                                                                                                  \
        else if constexpr (HEADDIM == 4)                                                                                                                                   \
        {                                                                                                                                                                  \
            return BINARY_DIM_SWITCH_4(VALUE, CONST_NAME, LAMBDA);                                                                                                         \
        }                                                                                                                                                                  \
        else if constexpr (HEADDIM == 8)                                                                                                                                   \
        {                                                                                                                                                                  \
            return BINARY_DIM_SWITCH_8(VALUE, CONST_NAME, LAMBDA);                                                                                                         \
        }                                                                                                                                                                  \
        else if constexpr (HEADDIM == 16)                                                                                                                                  \
        {                                                                                                                                                                  \
            return BINARY_DIM_SWITCH_16(VALUE, CONST_NAME, LAMBDA);                                                                                                        \
        }                                                                                                                                                                  \
        else if constexpr (HEADDIM == 32)                                                                                                                                  \
        {                                                                                                                                                                  \
            return BINARY_DIM_SWITCH_32(VALUE, CONST_NAME, LAMBDA);                                                                                                        \
        }                                                                                                                                                                  \
        else if constexpr (HEADDIM == 64)                                                                                                                                  \
        {                                                                                                                                                                  \
            return BINARY_DIM_SWITCH_64(VALUE, CONST_NAME, LAMBDA);                                                                                                        \
        }                                                                                                                                                                  \
        else if constexpr (HEADDIM == 128)                                                                                                                                 \
        {                                                                                                                                                                  \
            return BINARY_DIM_SWITCH_128(VALUE, CONST_NAME, LAMBDA);                                                                                                       \
        }                                                                                                                                                                  \
        else                                                                                                                                                               \
        {                                                                                                                                                                  \
            static_assert(HEADDIM == 2 || HEADDIM == 4 || HEADDIM == 8 || HEADDIM == 16 || HEADDIM == 32 || HEADDIM == 64 || HEADDIM == 128, "Unsupported HEADDIM value"); \
            return -1; \
        }                                                                                                                                                                  \
    }()


/**
 * @brief Statically calculate the index for the given outer dimension
 */
template < int Headdim>
__forceinline__ __device__ int get_col_idx(int dim);

template <>
__forceinline__ __device__ int get_col_idx<32>(int dim) {
    if (dim < 0 || dim >= 32) {
        return -1;
    }
    switch (dim) {
        case 0: return 0;
        case 1: return 1;
        case 2: return 0;
        case 3: return 1;
        case 4: return 0;
        case 5: return 1;
        case 6: return 0;
        case 7: return 1;
        case 8: return 2;
        case 9: return 3;
        case 10: return 2;
        case 11: return 3;
        case 12: return 2;
        case 13: return 3;
        case 14: return 2;
        case 15: return 3;
        case 16: return 4;
        case 17: return 5;
        case 18: return 4;
        case 19: return 5;
        case 20: return 4;
        case 21: return 5;
        case 22: return 4;
        case 23: return 5;
        case 24: return 6;
        case 25: return 7;
        case 26: return 6;
        case 27: return 7;
        case 28: return 6;
        case 29: return 7;
        case 30: return 6;
        case 31: return 7;
        default: return -1;
    }
}

template <>
__forceinline__ __device__ int get_col_idx<64>(int dim) {
    if (dim < 0 || dim >= 64) {
        return -1;
    }
    switch (dim) {
        case 0: return 0;
        case 1: return 1;
        case 2: return 0;
        case 3: return 1;
        case 4: return 0;
        case 5: return 1;
        case 6: return 0;
        case 7: return 1;
        case 8: return 2;
        case 9: return 3;
        case 10: return 2;
        case 11: return 3;
        case 12: return 2;
        case 13: return 3;
        case 14: return 2;
        case 15: return 3;
        case 16: return 4;
        case 17: return 5;
        case 18: return 4;
        case 19: return 5;
        case 20: return 4;
        case 21: return 5;
        case 22: return 4;
        case 23: return 5;
        case 24: return 6;
        case 25: return 7;
        case 26: return 6;
        case 27: return 7;
        case 28: return 6;
        case 29: return 7;
        case 30: return 6;
        case 31: return 7;
        case 32: return 8;
        case 33: return 9;
        case 34: return 8;
        case 35: return 9;
        case 36: return 8;
        case 37: return 9;
        case 38: return 8;
        case 39: return 9;
        case 40: return 10;
        case 41: return 11;
        case 42: return 10;
        case 43: return 11;
        case 44: return 10;
        case 45: return 11;
        case 46: return 10;
        case 47: return 11;
        case 48: return 12;
        case 49: return 13;
        case 50: return 12;
        case 51: return 13;
        case 52: return 12;
        case 53: return 13;
        case 54: return 12;
        case 55: return 13;
        case 56: return 14;
        case 57: return 15;
        case 58: return 14;
        case 59: return 15;
        case 60: return 14;
        case 61: return 15;
        case 62: return 14;
        case 63: return 15;
        default: return -1;
    }
}