#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <iostream>
#include <chrono>
#include <tuple>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/layout/layout.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/print_error.hpp>
#include <cutlass/util/GPU_Clock.hpp>
#include <cutlass/util/helper_cuda.hpp>

#include <cuda_runtime.h>
#include "utils.h"

enum class KernelType
{
    CHUNK_STATE_FWD,
    // CHUNK_STATE_BWD,
    QUERY_STATE_FWD,
    // QUERY_STATE_BWD,
};

/**
 * @brief Get the layout of the State Chunk Forward kernel.
 *
 * @param headdim The dimension of each head.
 * @param deg The degree of the polynomial approximation used.
 * @return A tuple containing:
 *         - expanded_dim: The expanded dimension.
 *         - BlockD: The fixed block size.
 *         - BlockT: The block size in the T dimension.
 */

template <KernelType kernel_type>
constexpr std::tuple<int, int, int> getLayout(int headdim, int deg)
{
    if constexpr (kernel_type == KernelType::CHUNK_STATE_FWD)
    {
        // we want BlockD to be large in CHUNK_STATE_FWD
        // NWarps = BlockD * BlockT / (16 * 16)
        if (deg == 2)
        {
            if (headdim == 32)
            {
                return std::make_tuple(528, 64, 16);
            }
            else if (headdim == 64)
            {
                return std::make_tuple(2080, 64, 16);
            }
        }
        else if (deg == 4)
        {
            if (headdim == 32)
            {
                return std::make_tuple(52360, 64, 16);
            }
            else if (headdim == 64)
            {
                return std::make_tuple(766480, 64, 16);
            }
        }
    }
    else if constexpr (kernel_type == KernelType::QUERY_STATE_FWD)
    {
        // we want BlockT to be large in QUERY_STATE_FWD
        // NWarps = BlockD * BlockT / (16 * 16)
        if (deg == 2)
        {
            if (headdim == 32)
            {
                return std::make_tuple(528, 16, 64);
            }
            else if (headdim == 64)
            {
                return std::make_tuple(2080, 16, 64);
            }
        }
        else if (deg == 4)
        {
            if (headdim == 32)
            {
                return std::make_tuple(52360, 16, 64);
            }
            else if (headdim == 64)
            {
                return std::make_tuple(766480, 16, 64);
            }
        }
    }
}

#define STATE_HEADDIM_SWITCH(HEADDIM, CONST_NAME, ...)              \
    [&] {                                                           \
        if (HEADDIM == 32)                                          \
        {                                                           \
            constexpr static int CONST_NAME = 32;                   \
            return __VA_ARGS__();                                   \
        }                                                           \
        else if (HEADDIM == 64)                                     \
        {                                                           \
            constexpr static int CONST_NAME = 64;                   \
            return __VA_ARGS__();                                   \
        }                                                           \
        else                                                        \
        {                                                           \
            throw std::runtime_error("Unsupported head dimension"); \
        }                                                           \
    }()

#define STATE_DEG_SWITCH(DEG, CONST_NAME, ...)                            \
    [&] {                                                                 \
        if (DEG == 2)                                                     \
        {                                                                 \
            constexpr static int CONST_NAME = 2;                          \
            return __VA_ARGS__();                                         \
        }                                                                 \
        else if (DEG == 4)                                                \
        {                                                                 \
            constexpr static int CONST_NAME = 4;                          \
            return __VA_ARGS__();                                         \
        }                                                                 \
        else                                                              \
        {                                                                 \
            throw std::runtime_error("Unsupported degree of similarity"); \
            return;                                                       \
        }                                                                 \
    }()

using namespace cute;

template <typename T, int Headdim_, int Deg_, KernelType kernel_type>
struct Power_Traits_Fwd
{

    using Element = T;
    using index_t = int64_t;
    using ElementAccum = float;
    using C_type = int;
    static constexpr int Headdim = Headdim_;
    static constexpr int Deg = Deg_;
    static constexpr int ExpandedDim = std::get<0>(getLayout<kernel_type>(Headdim, Deg));
    static constexpr int BlockD = std::get<1>(getLayout<kernel_type>(Headdim, Deg));
    static constexpr int BlockT = std::get<2>(getLayout<kernel_type>(Headdim, Deg));
    static constexpr int NWarps = BlockD * BlockT / (16 * 16);
    static constexpr int NThreads = NWarps * cutlass::NumThreadsPerWarp;
    static constexpr int NumBlockD = (ExpandedDim + BlockD - 1) / BlockD;

    // ==============================MMA======================================
    using MMA_Atom = std::conditional_t<
        std::is_same_v<Element, cutlass::half_t>,
        cute::MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        cute::MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;

    using TiledMma = TiledMMA<
        MMA_Atom,
        Layout<Shape<Int<NWarps>, _1, _1>>,
        Tile<Int<16 * NWarps>, _16, _16>>;

    // =============================== Smem Copy Atom ================================
    #if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 750
        using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
        using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, Element>;
    #else
        using SmemCopyAtom = Copy_Atom<DefaultCopy, Element>;
        using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, Element>;
    #endif

    // =============================== K ================================
    // See SmemLayoutK.pdf for the layout
    constexpr static int BlockDimK = 32;
    constexpr static int BlockRowsK = 512 / BlockDimK;
    using SmemLayoutAtomK = decltype(composition(Swizzle<3, 3, 3>{},
                                                 Layout<Shape<Shape<_2, Int<BlockRowsK / 2>>, Shape<_4, Int<BlockDimK / 4>>>,
                                                        Stride<Stride<_4, Int<BlockDimK * 2>>, Stride<_1, _8>>>{}));

    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{}, make_shape(Int<BlockT>{}, Int<Headdim>{})));

    // K transpose
    using SmemLayoutKt = decltype(composition(SmemLayoutK{}, make_layout(Shape<Int<Headdim>, Int<BlockT>>{},
                                                                         GenRowMajor{})));

    // We load K in 8 byte chunks, 4 elements per thread
    static constexpr int GmemElementsPerLoad = 4;
    static constexpr int GmemThreadsPerRowK = BlockDimK / GmemElementsPerLoad;
    // Tile layout for copying K from global memory to shared memory
    using GmemCopyTileK = decltype(make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint64_t>, Element>{},
        Layout<Shape<Int<NThreads / GmemThreadsPerRowK>, Int<GmemThreadsPerRowK>>,
               Stride<Int<GmemThreadsPerRowK>, _1>>{},
        Layout<Shape<_1, Int<GmemElementsPerLoad>>>{}));

    // Here we define the copy_atom for loading 8 elements at a time.
    using SmemCopyAtomK = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<sizeof(Element) * 8 * 8>, Element>;

    // =============================== V ================================
    // To avoid read bank conflict when doing matmul of phi(K)^T @ V, we need to swizzle sV.
    constexpr static int BlockDimV = Headdim % 64 == 0 ? 64 : 32;
    constexpr static int kSwizzleBV = Headdim == 32 ? 2 : 3;
    using SmemLayoutAtomV = decltype(composition(Swizzle<kSwizzleBV, 3, 3>{},
                                                 Layout<Shape<_8, Int<BlockDimV>>,
                                                        Stride<Int<BlockDimV>, _1>>{}));

    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        make_shape(Int<BlockT>{}, Int<Headdim>{})));

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    using SmemCopyAtomV = Copy_Atom<SM75_U16x8_LDSM_T, Element>;
#else
    using SmemCopyAtomV = Copy_Atom<DefaultCopy, Element>;
#endif

    // V transpose
    using SmemLayoutVt = decltype(composition(SmemLayoutV{}, make_layout(Shape<Int<Headdim>, Int<BlockT>>{}, GenRowMajor{})));
    // This is only to use the layout to create registers
    using SmemLayoutVtNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVt{}));

    // Use vectorized 128-bit mem load for V, which means 8 half_t/bfloat16_t elements per thread per load
    static constexpr int GmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element); // 8
    static_assert(Headdim % GmemElemsPerLoad == 0, "HeadDim must be divisible by GmemElemsPerLoad");

    // Tile layout for copying V
    static constexpr int GmemThreadsPerRowV = BlockDimV / GmemElemsPerLoad;
    using GmemCopyTileV = decltype(make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>{},
        Layout<Shape<Int<NThreads / GmemThreadsPerRowV>, Int<GmemThreadsPerRowV>>,
               Stride<Int<GmemThreadsPerRowV>, _1>>{},
        Layout<Shape<_1, _8>>{}));

    // =============================== S ================================
    using SmemLayoutS = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        Shape<Int<BlockD>, Int<Headdim>>{},
        GenRowMajor{}));

    // TODO (sean): use stmatrix?
    using SmemCopyAtomS = Copy_Atom<DefaultCopy, Element>;

    using GmemCopyTileS = decltype(make_tiled_copy(
        Copy_Atom<DefaultCopy, Element>{},
        Layout<Shape<Int<NThreads / Headdim>, Shape<Int<Headdim>>>,
               Stride<Int<Headdim>, _1>>{},
        Layout<Shape<_1, _1>>{})); // Tile Size: (NThreads / Headdim) x Headdim

    // =============================== Smem Size ================================
    static constexpr int KSmemSize = size(SmemLayoutK{}) * sizeof(Element);
    static constexpr int VSmemSize = size(SmemLayoutV{}) * sizeof(Element);
    static constexpr int SSmemSize = size(SmemLayoutS{}) * sizeof(Element);
    static constexpr int SmemSize = KSmemSize + VSmemSize + SSmemSize;
};

template <typename Kernel_traits, typename TK, typename TV, typename TS>
__global__ static __launch_bounds__(Kernel_traits::NWarps *cutlass::NumThreadsPerWarp, 1) void state_expand_kernel(__grid_constant__ const TK * const K, __grid_constant__ const TV * const V, __grid_constant__ TS * const S, __grid_constant__ const int chunk_size)
{
    using namespace cute;

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    using multiindex_t = int32_t;
    using C_type = int32_t;

    constexpr int BlockD = Kernel_traits::BlockD;
    constexpr int BlockT = Kernel_traits::BlockT;
    constexpr int Headdim = Kernel_traits::Headdim;
    constexpr int Deg = Kernel_traits::Deg;
    constexpr int NThreads = Kernel_traits::NThreads;
    constexpr bool ExpandedDim = Kernel_traits::ExpandedDim;
    constexpr int NumBlockD = Kernel_traits::NumBlockD;

    extern __shared__ char smem_[];

    const int tid = threadIdx.x;
    const int warp_id = tid / cutlass::NumThreadsPerWarp;
    const int dim_id = blockIdx.x;
    int kblockT = 0;
    const int numKBlockT = (chunk_size + BlockT - 1) / BlockT;

    //=============== Gloabl memory tensors =================
    Tensor gK = make_tensor(
        make_gmem_ptr(const_cast<Element *>(reinterpret_cast<const Element *>(K))),
        Layout<Shape<Int<BlockT>, Int<Headdim>>,
            Stride<Int<Headdim>, _1>>{});

    Tensor gV = make_tensor(
        make_gmem_ptr(const_cast<Element *>(reinterpret_cast<const Element *>(V))),
        Layout<Shape<Int<BlockT>, Int<Headdim>>,
            Stride<Int<Headdim>, _1>>{});

    const index_t S_offset = dim_id * BlockD * Headdim;
    Tensor gS = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element *>(S) + S_offset),
        Layout<Shape<Int<BlockD>, Int<Headdim>>,
               Stride<Int<Headdim>, _1>>{});

    // ================ Shared memory tensors =================
    Tensor sK = make_tensor(
        make_smem_ptr(reinterpret_cast<Element *>(smem_)),
        typename Kernel_traits::SmemLayoutK{});

    Tensor sV = make_tensor(
        sK.data() + size(sK),
        typename Kernel_traits::SmemLayoutV{});

    Tensor sVtNoSwizzle = make_tensor(
        sV.data(),
        typename Kernel_traits::SmemLayoutVtNoSwizzle{});

    Tensor sS = make_tensor(
        sV.data() + size(sV),
        typename Kernel_traits::SmemLayoutS{});

    // ================ MMA =================
    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);
    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<BlockD>, Int<Headdim>>{});
    clear(acc_o);

    // ================ Copy Atoms =================
    typename Kernel_traits::GmemCopyTileK gmem_tiled_copy_K;
    typename Kernel_traits::GmemCopyTileV gmem_tiled_copy_V;
    typename Kernel_traits::GmemCopyTileS gmem_tiled_copy_S;
    auto smem_tiled_copy_Vt = make_tiled_copy_B( // the ldmatrix call
        typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto gmem_thr_copy_K = gmem_tiled_copy_K.get_thread_slice(tid);
    auto gmem_thr_copy_V = gmem_tiled_copy_V.get_thread_slice(tid);
    auto gmem_thr_copy_S = gmem_tiled_copy_S.get_thread_slice(tid);
    auto smem_thr_copy_Vt = smem_tiled_copy_Vt.get_thread_slice(tid);

    Tensor tKgK = gmem_thr_copy_K.partition_S(gK);
    Tensor tKsK = gmem_thr_copy_K.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_V.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_V.partition_D(sV);
    Tensor tSgS = gmem_thr_copy_S.partition_S(gS);
    Tensor tSsS = gmem_thr_copy_S.partition_D(sS);
    Tensor tSsVt = smem_thr_copy_Vt.partition_S(sVtNoSwizzle);

    // ================ Predicates =================
    Tensor cS = make_identity_tensor(make_shape(size<0>(sS), size<1>(sS)));
    Tensor tScS = gmem_thr_copy_S.partition_D(cS);

    // ================ Loaders and Bumpers =================
    auto load_K = [&]()
    {
        cute::copy(gmem_tiled_copy_K, tKgK, tKsK);
    };
    auto bump_gK = [&]()
    {
        tKgK.data() = tKgK.data() + int(BlockT * Headdim);
    };
    auto load_V = [&]()
    {
        cute::copy(gmem_tiled_copy_V, tVgV, tVsV);
    };
    auto bump_gV = [&]()
    {
        tVgV.data() = tVgV.data() + int(BlockT * Headdim);
    };
    auto save_S = [&]()
    {
        state_kernel::copy</*Is_even_MN=*/false>(gmem_tiled_copy_S, tSsS, tSgS, tScS, ExpandedDim - dim_id * BlockD * Headdim);
    };

    // Don't worry about double buffering, this code is only for experiment
    // the actual kernel already does double buffering
    load_K();
    bump_gK();
    cute::cp_async_fence();

    // ================ Main Loop =================
    for (; kblockT < numKBlockT; kblockT++)
    {
        load_V();
        bump_gV();
        cute::cp_async_fence();

        // Wait for K
        cute::cp_async_wait<1>();
        __syncthreads();

        // Expand states
        // We are not going to use the sPKt, it's just for providing shape info
        Tensor sPKt = make_tensor(
            make_smem_ptr(reinterpret_cast<Element *>(smem_)),
            Shape<Int<BlockD>, Int<BlockT>>{});
        Tensor tSrPKt = thr_mma.partition_fragment_A(sPKt); // ((2, 2, 2), TILE_D, TILE_T)
        // ================ Expansion =================
        // This is where we should read from sK, and do some computation
        // and put the computation results on sPKt, and subsequently sPKt should be
        // used directly in the MMA
        

        // Start loading next K
        __syncthreads();
        if (kblockT < numKBlockT - 1)
        {
            load_K();
            bump_gK();
        }
        cute::cp_async_fence();

        // Wait for V
        cute::cp_async_wait<1>();
        __syncthreads();

        // Do matmul

        Tensor tSrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);
        state_kernel::gemm_rs(acc_o, tSrPKt, tSrVt, tSsVt, tiled_mma, smem_tiled_copy_Vt, smem_thr_copy_Vt);
        __syncthreads();
    }

    // write S back to smem
    auto smem_tiled_copy_S = make_tiled_copy_C(
        typename Kernel_traits::SmemCopyAtomS{},
        tiled_mma);
    auto smem_thr_copy_S = smem_tiled_copy_S.get_thread_slice(tid);
    Tensor rS = state_kernel::convert_type<Element>(acc_o);
    Tensor taccSrS = smem_thr_copy_S.retile_S(rS);
    Tensor taccSsS = smem_thr_copy_S.partition_D(sS);
    cute::copy(smem_tiled_copy_S, taccSrS, taccSsS);

    // write S back to gmem
    save_S();
}

template <typename Element, typename TK, typename TV, typename TS, KernelType kernel_type>
void run_state_expand(int T, int p, int d, int D,
                      TK const *K,
                      TV const *V,
                      TS *S,
                      cudaStream_t stream = 0)
{
    using namespace cute;

    STATE_DEG_SWITCH(p, Deg, [&]
                     { STATE_HEADDIM_SWITCH(d, Headdim, [&]
                                            {
            using Power_Traits = Power_Traits_Fwd<Element, Headdim, Deg, kernel_type>;
            constexpr int NWarps = Power_Traits::NWarps;
            constexpr int NumBlockD = Power_Traits::NumBlockD;
            constexpr int SmemSize = Power_Traits::SmemSize;

            if constexpr (kernel_type == KernelType::CHUNK_STATE_FWD) {
                dim3 grid(NumBlockD);
                dim3 block(NWarps * cutlass::NumThreadsPerWarp);
                state_expand_kernel<Power_Traits><<<grid, block, SmemSize, stream>>>
                    (K, V, S, T);
            } }); });
}

int main(int argc, char **argv)
{
    int T = 65536;
    int p = 4;
    int d = 32;
    const KernelType Kernel_type = KernelType::CHUNK_STATE_FWD;

    auto [D, BlockD, BlockT] = getLayout<Kernel_type>(d, p);

    if (argc >= 2)
        sscanf(argv[1], "%d", &T);
    if (argc >= 3)
        sscanf(argv[2], "%d", &p);
    if (argc >= 4)
        sscanf(argv[3], "%d", &d);

    using Element = cutlass::bfloat16_t;

    std::cout << "T = " << T << std::endl;
    std::cout << "p = " << p << std::endl;
    std::cout << "d = " << d << std::endl;
    std::cout << "D = " << D << std::endl;

    cute::device_init(0);

    thrust::host_vector<Element> h_K(T * d);
    thrust::host_vector<Element> h_V(T * d);
    thrust::host_vector<Element> h_S(D * d);

    for (int j = 0; j < T * d; ++j)
        h_K[j] = static_cast<Element>(2 * (rand() / double(RAND_MAX)) - 1);
    for (int j = 0; j < T * d; ++j)
        h_V[j] = static_cast<Element>(2 * (rand() / double(RAND_MAX)) - 1);
    for (int j = 0; j < D * d; ++j)
        h_S[j] = static_cast<Element>(2 * (rand() / double(RAND_MAX)) - 1);

    thrust::device_vector<Element> d_K = h_K;
    thrust::device_vector<Element> d_V = h_V;
    thrust::device_vector<Element> d_S = h_S;

    double gflops = (2.0 * D * d * T) * 1e-9;

    const int timing_iterations = 100;
    GPU_Clock timer;

    // Run once
    run_state_expand<Element, Element, Element, Element, Kernel_type>(T, p, d, D,
                                                                   d_K.data().get(),
                                                                   d_V.data().get(),
                                                                   d_S.data().get());
    CUTE_CHECK_LAST();
    thrust::host_vector<Element> cute_result = d_S;

    // Timing iterations
    timer.start();
    for (int i = 0; i < timing_iterations; ++i)
    {
        run_state_expand<Element, Element, Element, Element, Kernel_type>(T, p, d, D,
                                                                    d_K.data().get(),
                                                                    d_V.data().get(),
                                                                    d_S.data().get());
    }
    double cute_time = timer.seconds() / timing_iterations;
    CUTE_CHECK_LAST();
    printf("CUTE_GEMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time, cute_time * 1000);

    return 0;
}