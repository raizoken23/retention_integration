/******************************************************************************
 * Copyright (c) 2024, Sean Zhang.
 ******************************************************************************/

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <iostream>

#include "static_switch.h"
#include "power.h"
#include "power_fwd_kernel.h"

// Determine if the architecture supports POWER and define a macro to handle parameter modifiers
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define ARCH_SUPPORTS_POWER
#define KERNEL_PARAM_MODIFIER __grid_constant__
#else
#define KERNEL_PARAM_MODIFIER
#endif

// Define a macro for unsupported architecture handling to centralize the error message
#define POWER_UNSUPPORTED_ARCH printf("FATAL: PowerAttention requires building with sm version sm80-sm90, but was built for < 8.0!");

// Use a macro to clean up kernel definitions
#define DEFINE_POWER_FORWARD_KERNEL(kernelName, ...) \
template<typename Kernel_traits, __VA_ARGS__> \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const Power_fwd_params params)

DEFINE_POWER_FORWARD_KERNEL(power_fwd_kernel, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Return_softmax, bool IsGating, bool FlashEquivalent, bool NormalSpace, int Deg) {
    #if defined(ARCH_SUPPORTS_POWER)
        power::compute_attn<Kernel_traits, Is_causal, Is_even_MN, Is_even_K, Return_softmax, IsGating, FlashEquivalent, NormalSpace, Deg>(params);
    #else
        POWER_UNSUPPORTED_ARCH
    #endif
}


template<typename Kernel_traits, int Deg, bool Is_causal>
void run_power_fwd(Power_fwd_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;

    // Work-around for gcc 7. It doesn't like nested BOOL_SWITCH.
    // https://github.com/kokkos/kokkos-kernels/issues/349
    // https://github.com/HazyResearch/flash-attention/issues/21

    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    // const bool return_softmax = params.p_ptr != nullptr;
    constexpr bool ReturnSoftmaxConst = false;
    const bool is_gating = params.log_g_q_ptr != nullptr;
    const bool flash_equivalent = params.flash_equivalent;
    const bool normal_space = params.normal_space;
    // std::cout << "returning softmax: " << return_softmax << std::endl;
    // std::cout << params << std::endl;
    EVEN_MN_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
                GATING_SWITCH(is_gating, IsGatingConst, [&] {
#ifdef FULL_BUILD
                    FLASH_EQUIVALENT_SWITCH(flash_equivalent, FlashEquivalentConst, [&] { 
#else
                    if (flash_equivalent) {
                        TORCH_CHECK(false, "flash_equivalent is not supported for this build, rebuild the package with FULL_BUILD=1 to enable it");
                    }
                    constexpr bool FlashEquivalentConst = false;
#endif
                        NORMAL_SPACE_SWITCH(normal_space, NormalSpaceConst, [&] {
                            if constexpr (!NormalSpaceConst || !FlashEquivalentConst) {
                                // Will only return softmax if dropout, to reduce compilation time.
                                // If not IsEvenKConst, we also set IsEvenMNConst to false to reduce number of templates.
                                // If return_softmax, set IsEvenMNConst to false to reduce number of templates
                                // If head dim > 128, set IsEvenMNConst to false to reduce number of templates
                                auto kernel = &power_fwd_kernel<Kernel_traits, Is_causal, IsEvenMNConst && IsEvenKConst, IsEvenKConst, ReturnSoftmaxConst, IsGatingConst, FlashEquivalentConst, NormalSpaceConst, Deg>;
                                if (smem_size >= 48 * 1024) {
                                    C10_CUDA_CHECK(cudaFuncSetAttribute(
                                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                                }
                                kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                                C10_CUDA_KERNEL_LAUNCH_CHECK();
                            } else {
                                TORCH_CHECK(false, "normal_space and flash_equivalent cannot both be true at the same time");
                            }
                        });
#ifdef FULL_BUILD
                    });
#endif
            });
        });
    });
}


template<typename T, int Deg, bool Is_causal>
void run_mha_fwd_hdim32(Power_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 32;
    run_power_fwd<Power_fwd_kernel_traits<Headdim, 128, 128, 4, false, T>, Deg, Is_causal>(params, stream);
}

// template<typename T, bool Is_causal>
// void run_mha_fwd_hdim48(Power_fwd_params &params, cudaStream_t stream) {
//     constexpr static int Headdim = 48;
//     run_power_fwd<Power_fwd_kernel_traits<Headdim, 128, 128, 4, false, T>, Is_causal>(params, stream);
// }

template<typename T, int Deg, bool Is_causal>
void run_mha_fwd_hdim64(Power_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
#ifdef DEBUG_POWER_FWD
    run_power_fwd<Power_fwd_kernel_traits<Headdim, 128, 64, 4, true, T>, Deg, Is_causal>(params, stream);
#else
    run_power_fwd<Power_fwd_kernel_traits<Headdim, 128, 128, 4, false, T>, Deg, Is_causal>(params, stream);
#endif
}
template<typename T, int Deg, bool Is_causal>
void run_mha_fwd_hdim96(Power_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 96;
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm8x = dprops->major == 8 && dprops->minor > 0;
    // For sm86 or sm89, 64 x 64 is the fastest for causal (because it's square),
    if (is_sm8x) {
        if constexpr(!Is_causal) {
            run_power_fwd<Power_fwd_kernel_traits<Headdim, 128, 64, 4, false, T>, Deg, Is_causal>(params, stream);
        } else {
            run_power_fwd<Power_fwd_kernel_traits<Headdim, 64, 64, 4, false, T>, Deg, Is_causal>(params, stream);
        }
    } else {
        run_power_fwd<Power_fwd_kernel_traits<Headdim, 128, 64, 4, false, T>, Deg, Is_causal>(params, stream);
    }
}

template<typename T, int Deg, bool Is_causal>
void run_mha_fwd_hdim128(Power_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm8x = dprops->major == 8 && dprops->minor > 0;
    // For sm86 or sm89, 64 x 64 is the fastest for causal (because it's square),
    // and 128 x 32 (48 KB smem) is the fastest for non-causal since we get 2 CTAs per SM.
    if (is_sm8x) {
        if constexpr(!Is_causal) {
            run_power_fwd<Power_fwd_kernel_traits<Headdim, 128, 32, 4, false, T>, Deg, Is_causal>(params, stream);
        } else {
            run_power_fwd<Power_fwd_kernel_traits<Headdim, 64, 64, 4, false, T>, Deg, Is_causal>(params, stream);
        }
    } else {
        run_power_fwd<Power_fwd_kernel_traits<Headdim, 128, 64, 4, false, T>, Deg, Is_causal>(params, stream);
    }
}

template<typename T, int Deg, bool Is_causal>
void run_mha_fwd_hdim160(Power_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 160;
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm8x = dprops->major == 8 && dprops->minor > 0;
    // For A100, H100, 128 x 32 is the fastest.
    // For sm86 or sm89, 64 x 64 is the fastest for causal (because it's square),
    // and 128 x 64 with 8 warps is the fastest for non-causal.
    if (is_sm8x) {
        if constexpr(!Is_causal) {
            run_power_fwd<Power_fwd_kernel_traits<Headdim, 128, 64, 8, false, T>, Deg, Is_causal>(params, stream);
        } else {
            run_power_fwd<Power_fwd_kernel_traits<Headdim, 64, 64, 4, false, T>, Deg, Is_causal>(params, stream);
        }
    } else {
        run_power_fwd<Power_fwd_kernel_traits<Headdim, 128, 32, 4, false, T>, Deg, Is_causal>(params, stream);
    }
}

template<typename T, int Deg, bool Is_causal>
void run_mha_fwd_hdim192(Power_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 192;
    run_power_fwd<Power_fwd_kernel_traits<Headdim, 128, 64, 8, false, T>, Deg, Is_causal>(params, stream);
}

template<typename T, int Deg, bool Is_causal>
void run_mha_fwd_hdim224(Power_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 224;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    if (max_smem_per_block >= 2 * Headdim * (128 + 2 * 64)) {  // 112 KB
        run_power_fwd<Power_fwd_kernel_traits<Headdim, 128, 64, 8, false, T>, Deg, Is_causal>(params, stream);
    } else {
        run_power_fwd<Power_fwd_kernel_traits<Headdim, 64, 64, 4, false, T>, Deg, Is_causal>(params, stream);
    }
}

template<typename T, int Deg, bool Is_causal>
void run_mha_fwd_hdim256(Power_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_sm, max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
    status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    // For A100, we want to run with 128 x 64 (128KB smem).
    // For H100 we want to run with 64 x 64 (96KB smem) since then we can get 2 CTAs per SM.
    if (max_smem_per_block >= 2 * Headdim * (128 + 2 * 64) && max_smem_per_sm < 4 * Headdim * (64 + 2 * 64)) {
        run_power_fwd<Power_fwd_kernel_traits<Headdim, 128, 64, 8, false, T>, Deg, Is_causal>(params, stream);
    } else {
        run_power_fwd<Power_fwd_kernel_traits<Headdim, 64, 64, 4, false, T>, Deg, Is_causal>(params, stream);
    }
}

