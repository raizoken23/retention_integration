#pragma once

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "kernel_traits.h"
#include "state.h"
#include "utils.h"
#include "sympow.h"

// #define UPDATE_STATE_BWD_DEBUG 1
#define DEBUG_THREAD_CSBWD (tid == 0 && kid == 0 && head_id == 0 && batch_id == 0 && chunk_id == 0)
namespace power_attention
{

    using namespace cute;


    template <typename Kernel_traits, bool IsEvenN>
    __global__ void __launch_bounds__(Kernel_traits::NWarps *cutlass::NumThreadsPerWarp, 1) update_state_kernel_bwd(__grid_constant__ const Update_state_bwd_params params)
    {
        using Element = typename Kernel_traits::Element;
        using ElementAccum = typename Kernel_traits::ElementAccum;
        using index_t = typename Kernel_traits::index_t;
        // Shape parameters
        constexpr int BlockD = Kernel_traits::BlockD;
        constexpr int BlockK = Kernel_traits::BlockK;
        constexpr int Headdim = Kernel_traits::Headdim;
        constexpr int PaddedExpandedDim = Kernel_traits::PaddedExpandedDim;
        constexpr int InnerBlock = Kernel_traits::InnerBlock;
        constexpr int OuterBlock = Kernel_traits::OuterBlock;
        constexpr bool DoubleBuffer = Kernel_traits::DoubleBuffer;

        // Shared memory.
        extern __shared__ char smem_[];

        // Thread index
        const int tid = threadIdx.x;

        // Block index
        const int kid = blockIdx.x;
        const int head_id = blockIdx.y;
        const int batch_id = blockIdx.z / params.num_chunks;
        const int chunk_id = blockIdx.z % params.num_chunks;

        // block id
        const int numBlockD = (PaddedExpandedDim + BlockD - 1) / BlockD;


        // Offsets
        const index_t k_offset = blockIdx.z * params.chunk_stride + kid * BlockK * params.seq_stride + head_id * params.head_stride;
        const index_t v_offset = blockIdx.z * params.chunk_stride_v + kid * BlockK * params.seq_stride_v + head_id * params.head_stride_v;
        const index_t dS_offset = batch_id * params.batch_stride_s + chunk_id * params.chunk_stride_s + head_id * params.head_stride_s;

        // K
        Tensor gK = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + k_offset),
            Shape<Int<BlockK>, Int<Headdim>>{},
            make_stride(params.seq_stride, _1{})); // (BlockK, Headdim)

        // V
        Tensor gV = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + v_offset),
            Shape<Int<BlockK>, Int<Headdim>>{},
            make_stride(params.seq_stride_v, _1{})); // (BlockK, Headdim)

        // dS
        Tensor gdS = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.dS_ptr) + dS_offset),
            Shape<Int<BlockD>, Int<Headdim>>{},
            make_stride(Int<Headdim>{}, _1{})); // (BlockD, Headdim)

        // dK
        Tensor gdK = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + k_offset),
            Shape<Int<BlockK>, Int<Headdim>>{},
            make_stride(params.seq_stride, _1{})); // (BlockK, Headdim)

        // dV
        Tensor gdV = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + v_offset),
            Shape<Int<BlockK>, Int<Headdim>>{},
            make_stride(params.seq_stride_v, _1{})); // (BlockK, Headdim)

        // Share memory tensors
        Tensor sK = make_tensor(
            make_smem_ptr(reinterpret_cast<Element *>(smem_)),
            typename Kernel_traits::SmemLayoutK{}); // (BlockK, Headdim)

        Tensor sV = make_tensor(
            make_smem_ptr(reinterpret_cast<Element *>(&(*sK.data()) + size(sK))),
            typename Kernel_traits::SmemLayoutV{}); // (BlockK, Headdim)
        Tensor sVt = make_tensor(
            sV.data(),
            typename Kernel_traits::SmemLayoutVt{}); // (Headdim, BlockK)
        Tensor sVtNoSwizzle = make_tensor(
            sV.data(),
            typename Kernel_traits::SmemLayoutVtNoSwizzle{});

        Tensor sdS = make_tensor(
            sV.data() + size(sV),
            typename Kernel_traits::SmemLayoutdS{}); // (BlockD, Headdim)
        Tensor sdSt = make_tensor(
            sdS.data(),
            typename Kernel_traits::SmemLayoutdSt{}); // (Headdim, BlockD)

        Tensor sPK = make_tensor(
            make_smem_ptr(reinterpret_cast<Element *>(&(*sdS.data()) + (DoubleBuffer ? 2 : 1) * size(sdS))),
            typename Kernel_traits::SmemLayoutPhiK{}); // (BlockK, BlockD)
        Tensor sPKt = make_tensor(
            sPK.data(),
            typename Kernel_traits::SmemLayoutPhiKt{}); // (BlockD, BlockK)

        Tensor sdPK = make_tensor(
            sPK.data() + size(sPK),
            typename Kernel_traits::SmemLayoutPhiK{}); // (BlockD, BlockK)


        typename Kernel_traits::GmemCopyTileK gmem_tiled_copy_K;
        typename Kernel_traits::GmemCopyTileV gmem_tiled_copy_V;
        typename Kernel_traits::GmemCopyTiledS gmem_tiled_copy_dS;
        auto gmem_thr_copy_K = gmem_tiled_copy_K.get_thread_slice(tid);
        auto gmem_thr_copy_V = gmem_tiled_copy_V.get_thread_slice(tid);
        auto gmem_thr_copy_dS = gmem_tiled_copy_dS.get_thread_slice(tid);

        Tensor tKgK = gmem_thr_copy_K.partition_S(gK);     // ((CPY_ATOM_I, CPY_ATOM_J), TILE_M, TILE_N)
        Tensor tKsK = gmem_thr_copy_K.partition_D(sK);     // ((CPY_ATOM_I, CPY_ATOM_J), TILE_M, TILE_N)
        Tensor tVgV = gmem_thr_copy_V.partition_S(gV);     // ((CPY_ATOM_I, CPY_ATOM_J), TILE_M, TILE_N)
        Tensor tVsV = gmem_thr_copy_V.partition_D(sV);     // ((CPY_ATOM_I, CPY_ATOM_J), TILE_M, TILE_N)
        Tensor tdSgdS = gmem_thr_copy_dS.partition_S(gdS); // ((CPY_ATOM_I, CPY_ATOM_J), TILING_D, 1, nBlockD)
        Tensor tdSsdS = gmem_thr_copy_dS.partition_D(sdS); // ((CPY_ATOM_I, CPY_ATOM_J), TILING_D, 1)

        // MMA
        typename Kernel_traits::TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_thread_slice(tid);

        auto smem_tiled_copy_PhiK = make_tiled_copy_A(
            typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
        auto smem_thr_copy_PhiK = smem_tiled_copy_PhiK.get_thread_slice(tid);
        auto smem_tiled_copy_dSt = make_tiled_copy_B(
            typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
        auto smem_thr_copy_dSt = smem_tiled_copy_dSt.get_thread_slice(tid);
        // Tensor tdVsPK = smem_thr_copy_PhiK.partition_S(sPK);
        Tensor tdVsdSt = smem_thr_copy_dSt.partition_S(sdSt);

        // MMA 
        auto smem_tiled_copy_V = make_tiled_copy_A(
            typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
        auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tid);
        auto smem_tiled_copy_dS = make_tiled_copy_B(
            typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
        auto smem_thr_copy_dS = smem_tiled_copy_dS.get_thread_slice(tid);
        Tensor tdPKsV = smem_thr_copy_V.partition_S(sV);
        Tensor tdPKsdS = smem_thr_copy_dS.partition_S(sdS);
        Tensor tdPKrV = thr_mma.partition_fragment_A(sV); // can we afford to have this persistent in register?

        Tensor acc_dv = partition_fragment_C(tiled_mma, Shape<Int<BlockK>, Int<Headdim>>{});
        Tensor acc_dk = partition_fragment_C(tiled_mma, Shape<Int<BlockK>, Int<Headdim>>{});

        // Loaders and bumpers
        auto load_dS = [&]() {
            Tensor cdS = make_identity_tensor(Shape<Int<BlockD>, Int<Headdim>>{});
            Tensor tdScdS = gmem_thr_copy_dS.partition_S(cdS);
            power_attention::copy</*Is_even_MN=*/false>(gmem_tiled_copy_dS, tdSgdS, tdSsdS, tdScdS, BlockD);
            tdSgdS.data() = tdSgdS.data() + index_t(BlockD * Headdim);
        };
        auto bump_sdS = [&](const int dblock) {
            if constexpr (DoubleBuffer) {
                tdSsdS.data() = tdSsdS.data() + (dblock % 2 == 0 ? size(sdS) : -size(sdS));
            }
        };
        auto load_K = [&]() {
            Tensor cK = make_identity_tensor(Shape<Int<BlockK>, Int<Headdim>>{});
            Tensor tKcK = gmem_thr_copy_K.partition_S(cK);
            power_attention::copy</*Is_even_MN=*/IsEvenN>(gmem_tiled_copy_K, tKgK, tKsK, tKcK, std::min(BlockK, params.chunk_seq_len));
        };
        auto load_V = [&]() {
            Tensor cV = make_identity_tensor(Shape<Int<BlockK>, Int<Headdim>>{});
            Tensor tVcV = gmem_thr_copy_V.partition_S(cV);
            power_attention::copy</*Is_even_MN=*/IsEvenN>(gmem_tiled_copy_V, tVgV, tVsV, tVcV, std::min(BlockK, params.chunk_seq_len));
        };

        power_attention::SymowStateExpanderBlockP2<Kernel_traits> sympow;
        
        // Steps:
        // In each iteration, do:
        // 1. load next dS, next ds
        // 2. expand K
        // 3. wait for prev dS
        // 4. matmul phi_K dS => acc_dv
        // 5. matmul V dS^T => acc_dpk
        // 6. wait for prev ds
        // 7. add 1 @ ds^T to acc_dpk
        // 8. backprop dK => acc_dk

        // Prologue
        clear(acc_dv);
        clear(acc_dk);
        // copy K
        load_K();

        // copy dS
        load_dS();
        bump_sdS(0);

        // copy V
        load_V();

        cute::cp_async_fence();

        // Main loop
        for (int d_block = 0, inner_bid = 0, outer_bid = 0; d_block < numBlockD; ++d_block)
        {
            if (DoubleBuffer && d_block < numBlockD - 1)
            {
                // load dS block
                load_dS();
                bump_sdS(d_block + 1);
            }
            cute::cp_async_fence();
            // wait for K, V, dS, ds
            cute::cp_async_wait<DoubleBuffer ? 1 : 0>();
            __syncthreads();

#ifdef UPDATE_STATE_BWD_DEBUG
            if (DEBUG_THREAD_CSBWD) {
                printf("d_block: %d, sdS: \n", d_block);
                Tensor sdS_now = make_tensor(sdS.data() + (DoubleBuffer ? (d_block % 2 == 0 ? 0 : size(sdS)) : 0), sdS.layout());
                print_tensor(sdS_now);
                printf("\n");
                printf("d_block: %d, sds: \n", d_block);
                Tensor sds_now = make_tensor(sds.data() + (DoubleBuffer ? (d_block % 2 == 0 ? 0 : size(sds)) : 0), sds.layout());
                print_tensor(sds_now);
                printf("\n");
                printf("d_block: %d, sK: \n", d_block);
                print_tensor(sK);
                printf("\n");
                printf("d_block: %d, sV: \n", d_block);
                print_tensor(sV);
                printf("\n");
            }
            __syncthreads();
#endif

            // expand state
            Tensor tdVrPK = sympow.expandState(sK, tiled_mma, inner_bid, outer_bid);

#ifdef UPDATE_STATE_BWD_DEBUG
            Tensor tdVsPK_ = thr_mma.partition_A(sPK);
            cute::copy(tdVrPK, tdVsPK_);
            __syncthreads();
            if (DEBUG_THREAD_CSBWD) {
                printf("d_block: %d, sPK: \n", d_block);
                print_tensor(sPK);
                printf("\n");
            }
            __syncthreads();
#endif

            // matmul for dv
            Tensor tdVrdSt = thr_mma.partition_fragment_B(sdSt);
            power_attention::gemm_rs(acc_dv, tdVrPK, tdVrdSt, tdVsdSt, tiled_mma, smem_tiled_copy_dSt, smem_thr_copy_dSt);
            if constexpr (DoubleBuffer) {
                tdVsdSt.data() = tdVsdSt.data() + (d_block % 2 == 0 ? size(sdSt) : -size(sdSt));
            }

            // matmul for dPK
            Tensor acc_dpk = partition_fragment_C(tiled_mma, Shape<Int<BlockK>, Int<BlockD>>{});
            
            Tensor tdPKrdS = thr_mma.partition_fragment_B(sdS);
            if (d_block > 0) {
                power_attention::gemm_rs(acc_dpk, tdPKrV, tdPKrdS, tdPKsdS, tiled_mma, smem_tiled_copy_dS, smem_thr_copy_dS);
            } else {
                power_attention::gemm(acc_dpk, tdPKrV, tdPKrdS, tdPKsV, tdPKsdS, tiled_mma, smem_tiled_copy_V, smem_tiled_copy_dS, smem_thr_copy_V, smem_thr_copy_dS);
            }
            if constexpr (DoubleBuffer) {
                tdPKsdS.data() = tdPKsdS.data() + int((d_block % 2 == 0 ? size(sdS) : -size(sdS)));
            } else {
                load_dS();
                cute::cp_async_fence();
            }

            // backprop dK
            sympow.graddQK<true>(sK, acc_dpk, acc_dk, tiled_mma, inner_bid, outer_bid);

            // bump inner_bid, outer_bid
            binfo_bump<OuterBlock, InnerBlock>(inner_bid, outer_bid);
        }

        // Epilogue
        // copy dV back to shared memory, here use sV because we are done with it
        auto smem_tiled_copy_dV = make_tiled_copy_C(
            Copy_Atom<DefaultCopy, Element>{},
            tiled_mma);
        auto smem_thr_copy_dV = smem_tiled_copy_dV.get_thread_slice(tid);
        Tensor rdV = power_attention::convert_type<Element>(acc_dv);
        Tensor cdV = make_identity_tensor(Shape<Int<BlockK>, Int<Headdim>>{});
        Tensor tdVcsdV = smem_thr_copy_dV.partition_S(cdV);
        Tensor taccdVrdV = smem_thr_copy_dV.retile_S(rdV);
        Tensor taccdVsdV = smem_thr_copy_dV.partition_D(sV);
        // don't need to check Is_even_MN because we won't write OOB elements
        // to gmem anyway
        power_attention::copy</*Is_even_MN=*/true>(smem_tiled_copy_dV, taccdVrdV, taccdVsdV, tdVcsdV, BlockK);
        __syncthreads();

#ifdef UPDATE_STATE_BWD_DEBUG
        if (DEBUG_THREAD_CSBWD) {
            printf("sdV: \n");
            print_tensor(sV);
            printf("\n");
        }
        __syncthreads();
#endif

        // copy dV back to global memory
        typename Kernel_traits::GmemCopyTiledV gmem_tiled_copy_dV;
        auto gmem_thr_copy_dV = gmem_tiled_copy_dV.get_thread_slice(tid);
        Tensor tdVsdV = gmem_thr_copy_dV.partition_S(sV);
        Tensor tdVgdV = gmem_thr_copy_dV.partition_D(gdV);
        Tensor tdVcdV = gmem_thr_copy_dV.partition_S(cdV);
        power_attention::copy</*Is_even_MN=*/IsEvenN>(gmem_tiled_copy_dV, tdVsdV, tdVgdV, tdVcdV, std::min(BlockK, params.chunk_seq_len - kid * BlockK));
        __syncthreads();

        // copy dK back to shared memory, reuse sK because we are done with it
        auto smem_tiled_copy_dK = make_tiled_copy_C(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
        auto smem_thr_copy_dK = smem_tiled_copy_dK.get_thread_slice(tid);
        Tensor cdK = make_identity_tensor(Shape<Int<BlockK>, Int<Headdim>>{});
        Tensor rdK = power_attention::convert_type<Element>(acc_dk);
        Tensor tdKcsdK = smem_thr_copy_dK.partition_S(cdK);
        Tensor taccdKrdK = smem_thr_copy_dK.retile_S(rdK);
        Tensor taccdKsdK = smem_thr_copy_dK.partition_D(sK);
        power_attention::copy</*Is_even_MN=*/true>(smem_tiled_copy_dK, taccdKrdK, taccdKsdK, tdKcsdK, BlockK);
        __syncthreads();

#ifdef UPDATE_STATE_BWD_DEBUG
        if (DEBUG_THREAD_CSBWD) {
            printf("sdK: \n");
            print_tensor(sK);
            printf("\n");
        }
        __syncthreads();
#endif

        // copy dK back to global memory
        typename Kernel_traits::GmemCopyTiledK gmem_tiled_copy_dK;
        auto gmem_thr_copy_dK = gmem_tiled_copy_dK.get_thread_slice(tid);
        Tensor tdKcdK = gmem_thr_copy_dK.partition_S(cdK);
        Tensor tdKsdK = gmem_thr_copy_dK.partition_S(sK);
        Tensor tdKgdK = gmem_thr_copy_dK.partition_D(gdK);
        power_attention::copy</*Is_even_MN=*/IsEvenN>(gmem_tiled_copy_dK, tdKsdK, tdKgdK, tdKcdK, std::min(BlockK, params.chunk_seq_len - kid * BlockK));
    }

} // namespace power_attention
