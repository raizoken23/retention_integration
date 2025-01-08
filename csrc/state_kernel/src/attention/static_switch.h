// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```

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

#ifdef FLASHATTENTION_DISABLE_DROPOUT
#define DROPOUT_SWITCH(COND, CONST_NAME, ...)     \
    [&] {                                         \
        constexpr static bool CONST_NAME = false; \
        return __VA_ARGS__();                     \
    }()
#else
#define DROPOUT_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_ALIBI
#define ALIBI_SWITCH(COND, CONST_NAME, ...)       \
    [&] {                                         \
        constexpr static bool CONST_NAME = false; \
        return __VA_ARGS__();                     \
    }()
#else
#define ALIBI_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_UNEVEN_K
#define EVENK_SWITCH(COND, CONST_NAME, ...)      \
    [&] {                                        \
        constexpr static bool CONST_NAME = true; \
        return __VA_ARGS__();                    \
    }()
#else
#define EVENK_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_SOFTCAP
#define SOFTCAP_SWITCH(COND, CONST_NAME, ...)     \
    [&] {                                         \
        constexpr static bool CONST_NAME = false; \
        return __VA_ARGS__();                     \
    }()
#else
#define SOFTCAP_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_LOCAL
#define LOCAL_SWITCH(COND, CONST_NAME, ...)       \
    [&] {                                         \
        constexpr static bool CONST_NAME = false; \
        return __VA_ARGS__();                     \
    }()
#else
#define LOCAL_SWITCH BOOL_SWITCH
#endif

#ifdef FAST_IS_CAUSAL
    #define CAUSAL_SWITCH(COND, CONST_NAME, ...) \
        [&] { \
            constexpr static bool CONST_NAME = FAST_IS_CAUSAL; \
            return __VA_ARGS__(); \
        }()
#else
    #define CAUSAL_SWITCH BOOL_SWITCH
#endif

#ifdef FAST_IS_FP16
    #if FAST_IS_FP16
        #define FP16_SWITCH(COND, ...) \
            [&] { \
                using elem_type = cutlass::half_t; \
                return __VA_ARGS__(); \
            }()
    #else
        #define FP16_SWITCH(COND, ...) \
            [&] { \
                using elem_type = cutlass::bfloat16_t; \
                return __VA_ARGS__(); \
            }()
    #endif
#else
    #define FP16_SWITCH(COND, ...)                     \
        [&] {                                          \
            if (COND)                                  \
            {                                          \
                using elem_type = cutlass::half_t;     \
                return __VA_ARGS__();                  \
            }                                          \
            else                                       \
            {                                          \
                using elem_type = cutlass::bfloat16_t; \
                return __VA_ARGS__();                  \
            }                                          \
        }()
#endif

#ifdef FAST_HEAD_DIM
    #define HEADDIM_SWITCH(HEADDIM, ...) \
        [&] { \
            constexpr static int kHeadDim = FAST_HEAD_DIM; \
            return __VA_ARGS__(); \
        }()
#else
    #define HEADDIM_SWITCH(HEADDIM, ...)                          \
        [&] {                                                     \
            if (HEADDIM <= 32)                                    \
            {                                                     \
                constexpr static int kHeadDim = 32;               \
                return __VA_ARGS__();                             \
            }                                                     \
            else if (HEADDIM <= 64)                               \
            {                                                     \
                constexpr static int kHeadDim = 64;               \
                return __VA_ARGS__();                             \
            }                                                     \
            else                                                  \
            {                                                     \
                TORCH_CHECK(false, "Unsupported head dimension"); \
                return;                                           \
            }                                                     \
        }()
#endif

#ifdef FAST_DEG
    #define DEG_SWITCH(DEG, ...) \
        [&] { \
            constexpr static int kDeg = FAST_DEG; \
            return __VA_ARGS__(); \
        }()
#else
    #define DEG_SWITCH(DEG, ...)                                    \
        [&] {                                                       \
            if (DEG == 1) {                                        \
                constexpr static int kDeg = 1;                     \
                return __VA_ARGS__();                              \
            }                                                      \
            else if (DEG == 2) {                                  \
                constexpr static int kDeg = 2;                     \
                return __VA_ARGS__();                              \
            }                                                      \
            else if (DEG == 3) {                                  \
                constexpr static int kDeg = 3;                     \
                return __VA_ARGS__();                              \
            }                                                      \
            else if (DEG == 4) {                                  \
                constexpr static int kDeg = 4;                     \
                return __VA_ARGS__();                              \
            }                                                      \
            else {                                                \
                TORCH_CHECK(false, "Unsupported degree value");    \
                return;                                           \
            }                                                     \
        }()
#endif

#ifdef FAST_GATING
    #define GATING_SWITCH(COND, CONST_NAME, ...) \
        [&] { \
            constexpr static bool CONST_NAME = FAST_GATING; \
            return __VA_ARGS__(); \
        }()
#else
    #define GATING_SWITCH BOOL_SWITCH
#endif

#ifdef FAST_FLASH_EQUIVALENT
    #define FLASH_EQUIVALENT_SWITCH(COND, CONST_NAME, ...) \
        [&] { \
            constexpr static bool CONST_NAME = FAST_FLASH_EQUIVALENT; \
            return __VA_ARGS__(); \
        }()
#else
    #define FLASH_EQUIVALENT_SWITCH BOOL_SWITCH
#endif

#ifdef FAST_NORMAL_SPACE
    #define NORMAL_SPACE_SWITCH(COND, CONST_NAME, ...) \
        [&] { \
            constexpr static bool CONST_NAME = FAST_NORMAL_SPACE; \
            return __VA_ARGS__(); \
        }()
#else
    #define NORMAL_SPACE_SWITCH BOOL_SWITCH
#endif

#ifdef FAST_IS_EVEN_K
    #define EVENK_SWITCH(COND, CONST_NAME, ...) \
        [&] { \
            constexpr static bool CONST_NAME = FAST_IS_EVEN_K; \
            return __VA_ARGS__(); \
        }()
#else
    #ifdef FLASHATTENTION_DISABLE_UNEVEN_K
        #define EVENK_SWITCH(COND, CONST_NAME, ...)      \
            [&] {                                        \
                constexpr static bool CONST_NAME = true; \
                return __VA_ARGS__();                    \
            }()
    #else
        #define EVENK_SWITCH BOOL_SWITCH
    #endif
#endif
