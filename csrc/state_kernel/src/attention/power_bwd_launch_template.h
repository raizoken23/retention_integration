/******************************************************************************
 * Copyright (c) 2024, Sean Zhang.
 ******************************************************************************/

#pragma once

#include <ATen/cuda/CUDAContext.h>

#include "static_switch.h"
#include "power.h"
#include "power_bwd_kernel.h"
#include "power_bwd_convert_kernel.h"

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
#define DEFINE_POWER_BACKWARD_KERNEL(kernelName, ...) \
template<typename Kernel_traits, __VA_ARGS__> \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const Power_bwd_params params)


DEFINE_POWER_BACKWARD_KERNEL(power_bwd_dk_dv_kernel, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool IsGating) {
    #if defined(ARCH_SUPPORTS_POWER)
        power::compute_dk_dv<Kernel_traits, Is_causal, Is_even_MN, Is_even_K, IsGating>(params);
    #else
        POWER_UNSUPPORTED_ARCH
    #endif
}

DEFINE_POWER_BACKWARD_KERNEL(power_bwd_dq_kernel, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool IsGating) {
    #if defined(ARCH_SUPPORTS_POWER)
        power::compute_dq<Kernel_traits, Is_causal, Is_even_MN, Is_even_K, IsGating>(params);
    #else
        POWER_UNSUPPORTED_ARCH   
    #endif
}

DEFINE_POWER_BACKWARD_KERNEL(power_bwd_dq_dk_dv_kernel, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool IsGating, bool FlashEquivalent, bool NormalSpace, int Deg) {
    #if defined(ARCH_SUPPORTS_POWER)
        power::compute_dq_dk_dv<Kernel_traits, Is_causal, Is_even_MN, Is_even_K, IsGating, FlashEquivalent, NormalSpace, Deg>(params);
    #else
        POWER_UNSUPPORTED_ARCH
    #endif
}

template<typename Kernel_traits>
__global__ void power_bwd_convert_dq_kernel(const Power_bwd_params params, const int nsplits) {
    power::convert_dQ<Kernel_traits>(params, nsplits);
}


template<typename Kernel_traits, int Deg>
void run_power_bwd(Power_bwd_params &params, cudaStream_t stream) {
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    const int num_n_block = (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
    int gridDimx = num_n_block;
    int max_smem_per_sm;
    int device;
    cudaGetDevice(&device);
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    if (params.deterministic) {
        int device, sm_count;
		C10_CUDA_CHECK(cudaGetDevice(&device));
		C10_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
        gridDimx = (sm_count + params.b * params.h - 1) / (params.b * params.h);
    }
    dim3 grid_m(num_m_block, params.b, params.h);
    dim3 grid_n(gridDimx, params.b, params.h);

    // std::cout << "grid_m: " << grid_m.x << ", " << grid_m.y << ", " << grid_m.z << std::endl;

    // We want to specialize to is_even_MN and not just is_even_M, since in the case where N is not
    // a multiple of kBlockN, we'll need to apply mask in the loop.
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_q % Kernel_traits::kBlockM == 0 && params.seqlen_k % Kernel_traits::kBlockN == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool is_gating = params.log_g_q_ptr != nullptr;
    const bool flash_equivalent = params.flash_equivalent;
    const bool normal_space = params.normal_space;
    constexpr static bool Is_causal = true; // hardcode to reduce number of templates
    // constexpr int smem_size_dk_dv = Kernel_traits::kSmemSize1colblock;
    // constexpr int smem_size_dq = Kernel_traits::kSmemSize1rowblock;

    const int smem_size_dq_dk_dv = is_gating ? Kernel_traits::kSmemSize1colblock_dq_dk_dv_gating : Kernel_traits::kSmemSize1colblock_dq_dk_dv_no_gating;

    TORCH_CHECK(smem_size_dq_dk_dv <= max_smem_per_sm, "Total amount of smem required for dq_dk_dv = ", smem_size_dq_dk_dv, " > max_smem_per_sm = ", max_smem_per_sm);

    // BOOL_SWITCH(params.is_causal, Is_causal, [&] {
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
                            auto dq_dk_dv_kernel = &power_bwd_dq_dk_dv_kernel<Kernel_traits, Is_causal, IsEvenMNConst && IsEvenKConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst, IsGatingConst, FlashEquivalentConst, NormalSpaceConst, Deg>;
                            if (smem_size_dq_dk_dv >= 48 * 1024)  {
                                C10_CUDA_CHECK(cudaFuncSetAttribute(
                                    dq_dk_dv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_dq_dk_dv));
                            }
                            dq_dk_dv_kernel<<<grid_n, Kernel_traits::kNThreads, smem_size_dq_dk_dv, stream>>>(params);
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

    auto kernel_dq = &power_bwd_convert_dq_kernel<Kernel_traits>;
    if (Kernel_traits::kSmemdQSize >= 48 * 1024)  {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_dq, cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel_traits::kSmemdQSize));
    }
    kernel_dq<<<grid_m, Kernel_traits::kNThreads, Kernel_traits::kSmemdQSize, stream>>>(params, !params.deterministic ? 1 : gridDimx);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename T, int Deg>
void run_mha_bwd_hdim32(Power_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 32;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    // std::cout << "max_smem_per_block: " << max_smem_per_block << std::endl;
    if (max_smem_per_block >= 100 * 1024) { // 100 KB
        run_power_bwd<Power_bwd_kernel_traits<Headdim, /*BlockM*/128, /*BlockN*/128, /*NWarps*/8, /*SdP_M*/4, /*dKV_N*/4, /*dQ_M*/4, true, false, T>, Deg>(params, stream);
    } else {  // 102 KB
        run_power_bwd<Power_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, true, false, T>, Deg>(params, stream);
    }
}

// template<typename T>
// void run_mha_bwd_hdim48(Power_bwd_params &params, cudaStream_t stream) {
//     constexpr static int Headdim = 48;
//     int device;
//     cudaGetDevice(&device);
//     int max_smem_per_block;
//     cudaError status_ = cudaDeviceGetAttribute(
//         &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
//     if (status_ != cudaSuccess) {
//       C10_CUDA_CHECK(status_);
//     }
//     if (max_smem_per_block >= 2 * ((3 * 128 + 2 * 128) * Headdim + 2 * 128 * 128)) { // 104 KB
//         run_power_bwd<Power_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 8, 8, true, false, T>>(params, stream);
//     } else {  // 96 KB
//         run_power_bwd<Power_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 8, 8, false, false, T>>(params, stream);
//     }
// }

template<typename T, int Deg>
void run_mha_bwd_hdim64(Power_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    // std::cout << "max_smem_per_block: " << max_smem_per_block << std::endl;
    if (max_smem_per_block >= 146 * 1024) {
        run_power_bwd<Power_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, /*Is_V_in_regs=*/true, /*No_double_buffer=*/false, T>, Deg>(params, stream);
    } else {
        run_power_bwd<Power_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, /*Is_V_in_regs=*/true, /*No_double_buffer=*/false, T>, Deg>(params, stream);
    }
}

template<typename T, int Deg>
void run_mha_bwd_hdim96(Power_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 96;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    if (max_smem_per_block >= 116 * 1024) {
        run_power_bwd<Power_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, true, false, T>, Deg>(params, stream);
    } else {
        run_power_bwd<Power_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, true, false, T>, Deg>(params, stream);
    }
}

template<typename T, int Deg>
void run_mha_bwd_hdim128(Power_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    if (max_smem_per_block >= 144 * 1024) {
        run_power_bwd<Power_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 2, false, false, T>, Deg>(params, stream);
    } else {
        run_power_bwd<Power_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, true, false, T>, Deg>(params, stream);
    }
}

template<typename T, int Deg>
void run_mha_bwd_hdim160(Power_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 160;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    if (max_smem_per_block >= 116 * 1024) {
        run_power_bwd<Power_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 4, 4, false, false, T>, Deg>(params, stream);
    } else {
        run_power_bwd<Power_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 4, 4, false, true, T>, Deg>(params, stream);
    }
}

template<typename T, int Deg>
void run_mha_bwd_hdim192(Power_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 192;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    if (max_smem_per_block >= 136 * 1024) {
        run_power_bwd<Power_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, false, T>, Deg>(params, stream);
    } else {
        run_power_bwd<Power_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, true, true, T>, Deg>(params, stream);
    }
}

template<typename T, int Deg>
void run_mha_bwd_hdim224(Power_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 224;
    run_power_bwd<Power_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 4, 4, false, false, T>, Deg>(params, stream);
}

template<typename T, int Deg>
void run_mha_bwd_hdim256(Power_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    if (max_smem_per_block >= 176 * 1024) {  // H100
        run_power_bwd<Power_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, false, T>, Deg>(params, stream);
    } else if (max_smem_per_block >= 144 * 1024) {  // A100, we don't do double buffering to save smem
        run_power_bwd<Power_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, true, T>, Deg>(params, stream);
    } else { // sm86 and sm89, max smem is 99 KB. Only works without dropout. V in regs and no double buffering.
        run_power_bwd<Power_bwd_kernel_traits<Headdim, 64, 32, 8, 4, 1, 2, true, true, T>, Deg>(params, stream);
    }
}
