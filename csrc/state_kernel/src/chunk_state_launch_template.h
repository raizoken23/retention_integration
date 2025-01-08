#pragma once

#include <utility>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "state.h"
#include "static_switch.h"
#include "kernel_traits.h"
#include "chunk_state_fwd_kernel.h"
#include "chunk_state_bwd_kernel.h"


template <typename Kernel>
inline void check_smem_size(Kernel &kernel, size_t smem_size) {
    int device_id;
    cudaGetDevice(&device_id);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    // Get the shared memory size per block limit
    size_t smem_per_block = prop.sharedMemPerBlock;
    size_t smem_per_sm = prop.sharedMemPerBlockOptin;

    if (smem_size > smem_per_block && smem_size < smem_per_sm) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    } else if (smem_size > smem_per_sm) {
        std::stringstream ss;
        ss << "Required shared memory size (" << smem_size << " bytes) exceeds device limit ("
           << smem_per_sm << " bytes)";
        throw std::runtime_error(ss.str());
    }
}


template <typename T, int Headdim, int Deg>
void run_compute_chunk_states(Chunk_state_params &params, cudaStream_t stream);

template <typename T, int Headdim, int Deg>
void run_compute_chunk_states_bwd(Chunk_state_bwd_params &params, cudaStream_t stream);

template <typename Kernel_traits>
void run_chunk_states_fwd_kernel_(Chunk_state_params &params, cudaStream_t stream)
{
    TORCH_CHECK(params.chunk_seq_len >= Kernel_traits::BlockK, "chunk_seq_len must be at least ", Kernel_traits::BlockK);

    constexpr int PaddedExpandedDim = Kernel_traits::PaddedExpandedDim;
    constexpr int Headdim = Kernel_traits::Headdim;
    constexpr int BlockD = Kernel_traits::BlockD;
    constexpr int OuterBlock = Kernel_traits::OuterBlock;
    constexpr int InnerBlock = Kernel_traits::InnerBlock;
    constexpr int NumOuterBlocks = Headdim / OuterBlock;
    constexpr int NumInnerBlocks = Headdim / InnerBlock;
    static_assert(InnerBlock % OuterBlock == 0, "InnerBlock must be divisible by OuterBlock");
    constexpr int NumBlockD = (InnerBlock / OuterBlock + NumOuterBlocks) * NumInnerBlocks / 2;
    static_assert(NumBlockD * BlockD == PaddedExpandedDim, "NumBlockD * BlockD must be equal to PaddedExpandedDim");

    auto grid = dim3(NumBlockD, params.num_heads, params.batch_size * params.num_chunks);
    const size_t input_smem_size = Kernel_traits::InputSmemSize;
    const size_t output_smem_size = Kernel_traits::OutputSmemSize;
    const size_t phi_k_smem_size = Kernel_traits::PhiKSmemSize;
#ifdef CHUNK_STATE_FWD_DEBUG
    const size_t smem_size = std::max(input_smem_size + phi_k_smem_size, output_smem_size);
#else
    const size_t smem_size = std::max(input_smem_size + params.return_phi ? phi_k_smem_size : 0, output_smem_size);
#endif
    auto kernel = &state_kernel::chunk_state_kernel_fwd<Kernel_traits>;

    // std::cout << "chunk_states_fwd smem_size: " << smem_size << std::endl;

    kernel<<<grid, Kernel_traits::NThreads, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename Kernel_traits>
void run_chunk_states_bwd_kernel_(Chunk_state_bwd_params &params, cudaStream_t stream)
{
    auto minimum_chunk_len = cute::size(typename Kernel_traits::GmemCopyTileK::Tiler_MN{}) / Kernel_traits::Headdim;
    TORCH_CHECK(params.chunk_seq_len >= minimum_chunk_len, "chunk_seq_len must be at least ", minimum_chunk_len);

    constexpr int BlockK = Kernel_traits::BlockK;
    const bool is_even_N = params.chunk_seq_len % BlockK == 0;
    const int NumBlockK = (params.chunk_seq_len + BlockK - 1) / BlockK;
    auto grid = dim3(NumBlockK, params.num_heads, params.batch_size * params.num_chunks);
    EVEN_MN_SWITCH(is_even_N, IsEvenN, [&] {
        auto kernel = &state_kernel::chunk_state_kernel_bwd<Kernel_traits, IsEvenN>;
#ifdef CHUNK_STATE_BWD_DEBUG
        const int smem_size = Kernel_traits::SmemSize + Kernel_traits::PhiKSmemSize;
#else
        const int smem_size = Kernel_traits::SmemSize + (params.return_phi ? Kernel_traits::PhiKSmemSize * 2 : 0);
#endif
        check_smem_size(*kernel, smem_size);
        kernel<<<grid, Kernel_traits::NThreads, smem_size, stream>>>(params);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });

}

template <typename T, int Headdim, int Deg>
void run_chunk_states_fwd_(Chunk_state_params &params, cudaStream_t stream)
{
    constexpr auto layout = UniversalLayout<KernelType::StateChunkFwd>(Headdim, Deg);
    constexpr auto expandedDim = std::get<0>(layout);
    constexpr auto BlockD = std::get<1>(layout);
    constexpr auto BlockT = std::get<2>(layout);
    constexpr auto NWarps = std::get<3>(layout);
    constexpr auto D_layout = BlockDLayout<KernelType::StateChunkFwd>(Headdim);
    constexpr auto InnerBlock = std::get<0>(D_layout);
    constexpr auto OuterBlock = std::get<1>(D_layout);
    constexpr auto PaddedExpandedDim = std::get<2>(D_layout);
    // static_assert(OuterBlock == NWarps, "OuterBlock must be equal to NWarps");
    
    run_chunk_states_fwd_kernel_<State_chunk_traits<T, Headdim, Deg, expandedDim, BlockD, BlockT, NWarps, OuterBlock, InnerBlock, PaddedExpandedDim, /*DoubleBuffer_=*/true>>(params, stream);
}

template <typename T, int Headdim, int Deg>
void run_compute_chunk_states_bwd_(Chunk_state_bwd_params &params, cudaStream_t stream)
{
    constexpr auto layout = UniversalLayout<KernelType::StateChunkBwd>(Headdim, Deg);
    constexpr auto expandedDim = std::get<0>(layout);
    constexpr auto BlockD = std::get<1>(layout);
    constexpr auto BlockT = std::get<2>(layout);
    constexpr auto NWarps = std::get<3>(layout);
    constexpr auto D_layout = BlockDLayout<KernelType::StateChunkBwd>(Headdim);
    constexpr auto InnerBlock = std::get<0>(D_layout);
    constexpr auto OuterBlock = std::get<1>(D_layout);
    constexpr auto PaddedExpandedDim = std::get<2>(D_layout);

    run_chunk_states_bwd_kernel_<Chunk_state_bwd_traits<T, Headdim, Deg, expandedDim, BlockD, BlockT, NWarps, OuterBlock, InnerBlock, PaddedExpandedDim, /*DoubleBuffer_=*/true>>(params, stream);
}

