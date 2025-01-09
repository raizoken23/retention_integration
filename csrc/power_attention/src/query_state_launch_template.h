#pragma once

#include <utility>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <iostream>

#include "state.h"
#include "static_switch.h"
#include "kernel_traits.h"
#include "query_state_fwd_kernel.h"
#include "query_state_bwd_kernel.h"

template <typename Kernel_traits>
void run_query_states_fwd_kernel_(Query_state_params &params, cudaStream_t stream)
{

    TORCH_CHECK(params.chunk_seq_len >= Kernel_traits::BlockQ, "chunk_seq_len must be at least ", Kernel_traits::BlockQ);

    constexpr int BlockQ = Kernel_traits::BlockQ;
    constexpr auto input_smem_size = Kernel_traits::InputSmemSize;
    constexpr auto output_smem_size = Kernel_traits::OutputSmemSize;
    constexpr auto phi_q_smem_size = Kernel_traits::PhiQSmemSize;
    constexpr auto rowmax_smem_size = Kernel_traits::RowmaxSmemSize;    
#ifdef QUERY_STATE_FWD_DEBUG
    const auto smem_size = std::max(input_smem_size + phi_q_smem_size + (params.fused ? output_smem_size : 0) + (params.has_rowmax ? rowmax_smem_size : 0), output_smem_size);
#else
    const auto smem_size = std::max(input_smem_size + (params.return_phi ? phi_q_smem_size : 0) + (params.fused ? output_smem_size : 0) + (params.has_rowmax ? rowmax_smem_size : 0), output_smem_size);
#endif

    const int numBlockQ = (params.chunk_seq_len + BlockQ - 1) / BlockQ;
    auto kernel = &power_attention::query_power_attention_fwd<Kernel_traits>;
    auto grid = dim3(numBlockQ, params.num_heads, params.batch_size * params.num_chunks);

    // std::cout << "return_phi: " << params.return_phi << std::endl;
    // std::cout << "query_states_fwd smem_size: " << smem_size << std::endl;

    kernel<<<grid, Kernel_traits::NThreads, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
};


template <typename Kernel_traits>
void run_query_states_bwd_kernel_dsdn_(Query_state_bwd_params &params, cudaStream_t stream)
{
    constexpr int BlockD = Kernel_traits::BlockD;
    constexpr int numBlockD = (Kernel_traits::PaddedExpandedDim + BlockD - 1) / BlockD;
    auto griddSdN = dim3(numBlockD, params.num_heads, params.batch_size * params.num_chunks);
    auto kernel_dSdN = &power_attention::query_state_bwd_dSdN<Kernel_traits>;
    constexpr auto rowmax_smem_size = (Kernel_traits::DoubleBuffer ? 2 : 1) * Kernel_traits::RowmaxSmemSize;
#ifdef QUERY_STATE_BWD_DEBUG
    const size_t input_smem_size = Kernel_traits::InputSmemSizedSdN + Kernel_traits::PhiQSmemSize + (params.has_rowmax ? rowmax_smem_size : 0);
#else
    const size_t input_smem_size = Kernel_traits::InputSmemSizedSdN + (params.has_rowmax ? rowmax_smem_size : 0);
#endif
    const size_t output_smem_size = Kernel_traits::OutputSmemSizedSdN;
    const size_t smem_size = std::max(input_smem_size, output_smem_size);

    // std::cout << "query_states_bwd_dsdn smem_size: " << smem_size << std::endl;
    // std::cout << "query_states_bwd_dsdn:\n" << params << std::endl;
    
    kernel_dSdN<<<griddSdN, Kernel_traits::NThreads, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename Kernel_traits>
void run_query_states_bwd_kernel_dsdn_dq_(Query_state_bwd_params &params, cudaStream_t stream)
{
    constexpr int BlockD = Kernel_traits::BlockD;
    int gridDimD = (Kernel_traits::PaddedExpandedDim + BlockD - 1) / BlockD;
    auto kernel_dSdNdQ = &power_attention::query_state_bwd_dSdNdQ<Kernel_traits>;
    constexpr auto rowmax_smem_size = (Kernel_traits::DoubleBuffer ? 2 : 1) * Kernel_traits::RowmaxSmemSize;
#ifdef QUERY_STATE_BWD_DEBUG
    const size_t smem_size = Kernel_traits::SmemSizedSdNdQ + Kernel_traits::PhiQSmemSize + (params.has_rowmax ? rowmax_smem_size : 0);
#else
    const size_t smem_size = Kernel_traits::SmemSizedSdNdQ + (params.has_rowmax ? rowmax_smem_size : 0);
#endif

    if (params.deterministic) {
        int device, sm_count;
        C10_CUDA_CHECK(cudaGetDevice(&device));
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
        gridDimD = (sm_count + params.batch_size * params.num_heads * params.num_chunks - 1) / (params.batch_size * params.num_heads * params.num_chunks);
    }
    
    auto griddSdNdQ = dim3(gridDimD, params.num_heads, params.batch_size * params.num_chunks);

    // std::cout << "query_states_bwd_dsdn_dq smem_size: " << smem_size << std::endl;

    kernel_dSdNdQ<<<griddSdNdQ, Kernel_traits::NThreads, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template <typename Kernel_traits>
void run_query_states_bwd_kernel_dq_(Query_state_bwd_params &params, cudaStream_t stream)
{
    // The size of chunk size needs to be larger than BlockQ for obvious reasons
    constexpr int BlockQ = Kernel_traits::BlockQ;
    TORCH_CHECK(params.chunk_seq_len >= BlockQ, "chunk_seq_len must >= BlockQ:", BlockQ);

    const int numBlockQ = (params.chunk_seq_len + BlockQ - 1) / BlockQ;
    auto griddQ = dim3(numBlockQ, params.num_heads, params.batch_size * params.num_chunks);
    auto kernel_dQ = &power_attention::query_state_bwd_dQ<Kernel_traits>;
    const size_t input_smem_size = Kernel_traits::InputSmemSizedQ;
    const size_t phi_q_smem_size = Kernel_traits::PhiQSmemSize;
    const size_t output_smem_size = Kernel_traits::OutputSmemSizedQ;
    const size_t rowmax_smem_size = Kernel_traits::RowmaxSmemSize;
    const size_t smem_size = std::max(input_smem_size + phi_q_smem_size + (params.has_rowmax ? rowmax_smem_size : 0), output_smem_size);

    // std::cout << "query_states_bwd_dq smem_size: " << smem_size << std::endl;

    kernel_dQ<<<griddQ, Kernel_traits::NThreads, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T, int Headdim, int Deg>
void run_compute_query_states_(Query_state_params &params, cudaStream_t stream)
{
    constexpr auto layout = UniversalLayout<KernelType::QueryStateFwd>(Headdim, Deg);
    constexpr auto expandedDim = std::get<0>(layout);
    constexpr auto BlockD = std::get<1>(layout);
    constexpr auto BlockT = std::get<2>(layout);
    constexpr auto NWarps = std::get<3>(layout);
    constexpr auto D_layout = BlockDLayout<KernelType::QueryStateFwd>(Headdim);
    constexpr auto InnerBlock = std::get<0>(D_layout);
    constexpr auto OuterBlock = std::get<1>(D_layout);
    constexpr auto PaddedExpandedDim = std::get<2>(D_layout);

    run_query_states_fwd_kernel_<Query_state_traits<T, Headdim, Deg, expandedDim, BlockD, BlockT, NWarps, OuterBlock, InnerBlock, PaddedExpandedDim, /*DoubleBuffer*/true>>(params, stream);
}


template <typename T, int Headdim, int Deg>
void run_compute_query_states_bwd_(Query_state_bwd_params &params, cudaStream_t stream)
{
    constexpr auto layout_dSdN = UniversalLayout<KernelType::QueryStateBwddSdN>(Headdim, Deg);
    constexpr auto layout_D_dSdN = BlockDLayout<KernelType::QueryStateBwddSdN>(Headdim);
    constexpr auto ExpandedDim_dSdN = std::get<0>(layout_dSdN);
    constexpr auto BlockD_dSdN = std::get<1>(layout_dSdN);
    constexpr auto BlockT_dSdN = std::get<2>(layout_dSdN);
    constexpr auto NWarps_dSdN = std::get<3>(layout_dSdN);
    constexpr auto InnerBlock_dSdN = std::get<0>(layout_D_dSdN);
    constexpr auto OuterBlock_dSdN = std::get<1>(layout_D_dSdN);
    constexpr auto PaddedExpandedDim_dSdN = std::get<2>(layout_D_dSdN);

    if constexpr (SPLIT_QSBWD) {
        run_query_states_bwd_kernel_dsdn_<Query_state_bwd_traits<T, Headdim, Deg, ExpandedDim_dSdN, BlockD_dSdN, BlockT_dSdN, NWarps_dSdN, OuterBlock_dSdN, InnerBlock_dSdN, PaddedExpandedDim_dSdN, /*DoubleBuffer_*/true, /*S_in_regs_*/true>>(params, stream);
        
        constexpr auto layout_dQ = UniversalLayout<KernelType::QueryStateBwddQ>(Headdim, Deg);
        constexpr auto layout_D_dQ = BlockDLayout<KernelType::QueryStateBwddQ>(Headdim);
        constexpr auto ExpandedDim_dQ = std::get<0>(layout_dQ);
        constexpr auto BlockD_dQ = std::get<1>(layout_dQ);
        constexpr auto BlockT_dQ = std::get<2>(layout_dQ);
        constexpr auto NWarps_dQ = std::get<3>(layout_dQ);
        constexpr auto InnerBlock_dQ = std::get<0>(layout_D_dQ);
        constexpr auto OuterBlock_dQ = std::get<1>(layout_D_dQ);
        constexpr auto PaddedExpandedDim_dQ = std::get<2>(layout_D_dQ);

        run_query_states_bwd_kernel_dq_<Query_state_bwd_traits<T, Headdim, Deg, ExpandedDim_dQ, BlockD_dQ, BlockT_dQ, NWarps_dQ, OuterBlock_dQ, InnerBlock_dQ, PaddedExpandedDim_dQ, true, true>>(params, stream);
    } else {
        run_query_states_bwd_kernel_dsdn_dq_<Query_state_bwd_traits<T, Headdim, Deg, ExpandedDim_dSdN, BlockD_dSdN, BlockT_dSdN, NWarps_dSdN, OuterBlock_dSdN, InnerBlock_dSdN, PaddedExpandedDim_dSdN, /*DoubleBuffer_*/true, /*S_in_regs_*/true>>(params, stream);
    }
}

