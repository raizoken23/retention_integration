#pragma once

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/layout/layout.h>
#include <cutlass/numeric_types.h>

#include "attention/kernel_traits.h"

using namespace cute;

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define OOB_CHECK(layout, elements_per_thread) \
    static_assert(decltype(size(layout{}))::value <= NThreads * elements_per_thread || decltype(size(layout{}))::value % (NThreads * elements_per_thread) == 0, \
    #layout " must be less than or equal to NThreads * " #elements_per_thread " or a multiple of NThreads * " #elements_per_thread);

template <typename T, int Headdim_, int Deg_, int ExpandedDim_, int BlockD_, int BlockT_, int NWarps_, int OuterBlock_, int InnerBlock_, int PaddedExpandedDim_, bool DoubleBuffer_ = true>
struct State_chunk_traits
{
    using Element = T;
    using index_t = int64_t;
    using ElementAccum = float;
    using C_type = int;
    #if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 750
        using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
        using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x4_LDSM_T, Element>;
    #else
        using SmemCopyAtom = Copy_Atom<DefaultCopy, Element>;
        using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, Element>;
    #endif
    using multiindex_t = int32_t;

    // Number of threads
    static constexpr int NWarps = NWarps_;
    static constexpr int NThreads = NWarps * cutlass::NumThreadsPerWarp;

    static_assert(Deg_ == 2 || Deg_ == 4, "Deg must be 2 or 4");
    static_assert(NWarps == 1 || NWarps == 2 || NWarps == 4 || NWarps == 8 || NWarps == 12 || NWarps == 16 || NWarps == 32, "NWarps must be 4, 8, 12, or 16");

    // size of D dim assigned to a CTA
    static constexpr int BlockD = BlockD_;
    static constexpr int Headdim = Headdim_;
    static constexpr int Deg = Deg_;
    static constexpr int ExpandedDim = ExpandedDim_;
    static constexpr int OuterBlock = OuterBlock_;
    static constexpr int InnerBlock = InnerBlock_;
    static constexpr int PaddedExpandedDim = PaddedExpandedDim_;
    // just heuristics, this is also the minimum chunk size
    static constexpr int BlockT = BlockT_;
    static constexpr int BlockK = BlockT_;
    // To see what the swizzle parameter means, see the following
    static constexpr int DoubleBuffer = DoubleBuffer_;

    // row size to reason about swizzle in shared memory
    static constexpr int BlockDimSmem = Headdim % 64 == 0 ? 64 : 32;

    static_assert(NThreads >= BlockD, "NThreads must be greater than or equal to BlockD");

    // =============================== K ================================
    // See SmemLayoutK.pdf for the layout
    constexpr static int BlockDimK = Headdim % 64 == 0 ? 64 : 32;
    using SmemLayoutAtomKV2 = decltype(
        composition(Swizzle<BlockDimK == 32 ? 3 : 4, 1, 5>{},
                    Layout<Shape<_16, Int<BlockDimK>>,
                           Stride<Int<BlockDimK>, _1>>{}));

    using SmemLayoutAtomKV8 = decltype(
        composition(Swizzle<BlockDimK == 32 ? 2 : 3, 3, 3>{},
                    Layout<Shape<_8, Int<BlockDimK>>,
                           Stride<Int<BlockDimK>, _1>>{}));

    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomKV8{}, make_shape(Int<BlockK>{}, Int<Headdim>{})));
    // using SmemLayoutK = Layout<Shape<Int<BlockK>, Int<Headdim>>, Stride<Int<Headdim>, _1>>;

    // K transpose
    using SmemLayoutKt = decltype(composition(SmemLayoutK{}, make_layout(Shape<Int<Headdim>, Int<BlockK>>{},
                                                                          GenRowMajor{})));
    // using SmemLayoutKt = decltype(tile_to_shape(SmemLayoutAtomKV8{}, make_shape(Int<Headdim>{}, Int<BlockK>{})));

    // We load K in 4 byte chunks, 2 elements per thread
    using CopyStruct = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    static constexpr int ElemPerThread = 8;
    static constexpr int GmemThreadsPerRowK = BlockDimK / ElemPerThread;
    // Tile layout for copying K from global memory to shared memory
    using GmemCopyTileK = decltype(make_tiled_copy(
        Copy_Atom<CopyStruct, Element>{},
        Layout<Shape<Int<NThreads / GmemThreadsPerRowK>, Int<GmemThreadsPerRowK>>,
               Stride<Int<GmemThreadsPerRowK>, _1>>{},
        make_layout(Shape<_1, Int<ElemPerThread>>{}, GenRowMajor{})));

    // Here we define the copy_atom for loading 8 elements at a time.
    using SmemCopyAtomK = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<sizeof(Element) * 8 * 8>, Element>;

    // Tile layout for loading K from shared memory to registers, 2 rows 4 columns at a time
    using SmemCopyTileK = decltype(make_tiled_copy(
        SmemCopyAtomK{},
        Layout<Shape<Int<NThreads>, _1>, Stride<_1, _1>>{},
        Layout<Shape<_2, _4>>{}));

    // =============================== Phi K ================================
    using SmemLayoutPhiK = Layout<Shape<Int<BlockK>, Int<BlockD>>, Stride<Int<BlockD>, _1>>;

    using SmemLayoutPhiKt = decltype(composition(SmemLayoutPhiK{}, make_layout(Shape<Int<BlockD>, Int<BlockK>>{}, GenRowMajor{})));

    constexpr static int PhiKThreadsPerRow = BlockD <= NThreads ? BlockD : NThreads;
    constexpr static int PhiKThreadsPerCol = BlockD <= NThreads ? NThreads / BlockD : 1;
    static_assert((BlockD <= NThreads && NThreads % BlockD == 0) || (NThreads < BlockD && BlockD % NThreads == 0), "PhiKThreadsPerRow and PhiKThreadsPerCol must be valid");
    using GmemCopyTilePhiK = decltype(make_tiled_copy(
        Copy_Atom<DefaultCopy, Element>{},
        Layout<Shape<Int<PhiKThreadsPerCol>, Int<PhiKThreadsPerRow>>,
               Stride<Int<PhiKThreadsPerRow>, _1>>{},
        Layout<Shape<_1, _1>>{}));

    // =============================== V ================================
    // To avoid read bank conflict when doing matmul of phi(K)^T @ V, we need to swizzle sV.
    constexpr static int BlockDimV = Headdim % 64 == 0 ? 64 : 32;
    constexpr static int kSwizzleBV = BlockDimV == 32 ? 2 : 3;
    using SwizzleAtomV8 = decltype(composition(Swizzle<kSwizzleBV, 3, 3>{},
                                              Layout<Shape<_8, Int<BlockDimV>>,
                                                     Stride<Int<BlockDimV>, _1>>{}));
    using SmemLayoutAtomV = SwizzleAtomV8;

    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        make_shape(Int<BlockK>{}, Int<Headdim>{})));


    #if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 750
        using SmemCopyAtomV = Copy_Atom<SM75_U16x8_LDSM_T, Element>;
    #else
        using SmemCopyAtomV = Copy_Atom<DefaultCopy, Element>;
    #endif

    // V transpose
    using SmemLayoutVt = decltype(composition(SmemLayoutV{}, make_layout(Shape<Int<Headdim>, Int<BlockK>>{}, GenRowMajor{})));
    // This is only to use the layout to create registers
    using SmemLayoutVtNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVt{}));

    // Use vectorized 128-bit mem load for V, which means 8 half_t/bfloat16_t elements per thread per load
    static constexpr int GmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element); // 8
    static_assert(Headdim % GmemElemsPerLoad == 0, "HeadDim must be divisible by GmemElemsPerLoad");
    
    // Tile layout for copying V
    static constexpr int GmemThreadsPerRowV = BlockDimV / 8;
    using GmemCopyTileV = decltype(make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>{},
        Layout<Shape<Int<NThreads / GmemThreadsPerRowV>, Int<GmemThreadsPerRowV>>,
               Stride<Int<GmemThreadsPerRowV>, _1>>{},
        Layout<Shape<_1, _8>>{}));

    // OOB_CHECK(SmemLayoutV, 8);

    // =============================== O ================================
    using SmemLayoutAtomO = SmemLayoutAtomV;
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<BlockD>, Int<Headdim>>{},
        GenRowMajor{}));

    // TODO (sean): use stmatrix?
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;

    static_assert((NThreads >= Headdim && NThreads % Headdim == 0) || (NThreads < Headdim && Headdim % NThreads == 0), "NThreads must be divisible by Headdim or Headdim must be divisible by NThreads");
    static constexpr int kGmemThreadsPerRowO = NThreads > Headdim ? Headdim : NThreads;
    
    using GmemThrLayoutO = Layout<Shape<Int<NThreads / kGmemThreadsPerRowO>, Int<kGmemThreadsPerRowO>>, Stride<Int<kGmemThreadsPerRowO>, _1>>;

    using GmemCopyTileO = decltype(make_tiled_copy(
        Copy_Atom<DefaultCopy, Element>{},
        GmemThrLayoutO{},
        Layout<Shape<_1, _1>>{})); // Tile Size: (NThreads / Headdim) x Headdim

    // OOB_CHECK(SmemLayoutO, 2);

    // =============================== N ================================
    using SmemLayoutN = Layout<
        Shape<Int<BlockD>>,
        Stride<_1>>;

    using GmemCopyTileN = decltype(make_tiled_copy(
        Copy_Atom<DefaultCopy, ElementAccum>{},
        Layout<Shape<Int<NThreads>>, Stride<_1>>{},
        Layout<Shape<_1>>{}));

    // OOB_CHECK(SmemLayoutN, 1);

    // =============================== Smem Size ================================
    // smem size to store V
    static constexpr int VSmemSize = size(SmemLayoutV{}) * sizeof(Element); // BlockK * Headdim
    // smem size to store K
    static constexpr int KSmemSize = (DoubleBuffer ? 2 : 1) * size(SmemLayoutK{}) * sizeof(Element); // (DoubleBuffer ? 2 : 1) * Headdim * BlockK
    static constexpr int PhiKSmemSize = size(SmemLayoutPhiK{}) * sizeof(ElementAccum); // BlockK * BlockD
    static constexpr int InputSmemSize = VSmemSize + KSmemSize;
    // smem size to store O
    static constexpr int OSmemSize = size(SmemLayoutO{}) * sizeof(Element); // BlockD * Headdim
    // smem size to store norm
    static constexpr int NSmemSize = size(SmemLayoutN{}) * sizeof(ElementAccum);
    static constexpr int OutputSmemSize = OSmemSize;


    // =============================== MMA ================================
    // TODO: We might be able to use SM80_16x8x8_F32BF16BF16F32_TN to use a smaller BlockT
    using MMA_ATOM = std::conditional_t<
        std::is_same_v<Element, cutlass::half_t>,
        cute::MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        cute::MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;

    using TiledMma = TiledMMA<
        MMA_ATOM,
        Layout<Shape<Int<NWarps>, _1, _1>>,
        Tile<Int<16 * NWarps>, _16, _16>>;
};


template <typename T, int Headdim_, int Deg_, int ExpandedDim_, int BlockD_, int BlockT_, int NWarps_, int OuterBlock_, int InnerBlock_, int PaddedExpandedDim_, bool DoubleBuffer_ = true, typename Base = State_chunk_traits<T, Headdim_, Deg_, ExpandedDim_, BlockD_, BlockT_, NWarps_, OuterBlock_, InnerBlock_, PaddedExpandedDim_, DoubleBuffer_>>
struct Update_state_bwd_traits : public Base
{
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    using C_type = typename Base::C_type;
    using multiindex_t = typename Base::multiindex_t;

    static constexpr int NWarps = Base::NWarps;
    static constexpr int NThreads = Base::NThreads;
    static constexpr int BlockD = Base::BlockD;
    static constexpr int BlockK = Base::BlockK;
    static constexpr int Headdim = Base::Headdim;
    static constexpr int PaddedExpandedDim = PaddedExpandedDim_;
    static constexpr bool DoubleBuffer = DoubleBuffer_;

    using SmemLayoutK = typename Base::SmemLayoutK;
    using SmemLayoutV = typename Base::SmemLayoutV;
    using SmemLayoutVt = typename Base::SmemLayoutVt;
    using SmemLayoutVtNoSwizzle = typename Base::SmemLayoutVtNoSwizzle;
    using SmemLayouts = typename Base::SmemLayoutN;
    using SmemLayoutAtomV = typename Base::SmemLayoutAtomV;
    #if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 750
        using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
        using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, Element>;
    #else
        using SmemCopyAtom = Copy_Atom<DefaultCopy, Element>;
        using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, Element>;
    #endif

    using SmemLayoutPhiK = typename Base::SmemLayoutPhiK;
    using SmemLayoutPhiKt = typename Base::SmemLayoutPhiKt;

    using SmemLayoutdS = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        Shape<Int<BlockD>, Int<Headdim>>{},
        GenRowMajor{}));
    using SmemLayoutdSt = decltype(composition(
        SmemLayoutdS{},
        make_layout(Shape<Int<Headdim>, Int<BlockD>>{},
                    GenRowMajor{})));

    using SmemLayoutdK = SmemLayoutK;
    using SmemLayoutdV = SmemLayoutV;

    using SmemLayoutds = Layout<Shape<Int<BlockD>>, Stride<_1>>;

    static constexpr int KSmemSize = size(SmemLayoutK{}) * sizeof(Element);
    static constexpr int VSmemSize = size(SmemLayoutV{}) * sizeof(Element);
    static constexpr int PhiKSmemSize = size(SmemLayoutPhiK{}) * sizeof(Element);
    static constexpr int dSSmemSize = (DoubleBuffer ? 2 : 1) * size(SmemLayoutdS{}) * sizeof(Element);
    static constexpr int dsSmemSize = (DoubleBuffer ? 2 : 1) * size(SmemLayoutds{}) * sizeof(ElementAccum);
    static constexpr int dKSmemSize = size(SmemLayoutdK{}) * sizeof(Element);
    static constexpr int dVSmemSize = size(SmemLayoutdV{}) * sizeof(Element);
    static constexpr int InputSmemSize = KSmemSize + VSmemSize + dSSmemSize;
    static constexpr int OutputSmemSize = dVSmemSize + dKSmemSize;
    static constexpr int SmemSize = std::max(InputSmemSize, OutputSmemSize);

    // input copy
    static constexpr int GmemThreadsPerRow_Element = Base::GmemThreadsPerRowK;
    static constexpr int ThreadsPerSwizzleRow_Element = Base::BlockDimSmem / Base::GmemElemsPerLoadK;
    static constexpr int Deg = Deg_;

    static_assert(Deg == 2 || Deg == 4, "Deg must be 2 or 4");

    using SmemCopyTiledK = decltype(make_tiled_copy(
    Copy_Atom<DefaultCopy, Element>{},
    Layout<Shape<Int<NThreads>, _1>, Stride<_1, _1>>{}));
    using GmemCopyTiledS = typename Base::GmemCopyTileK;
    using GmemCopyTileds = typename Base::GmemCopyTileN;
    // output copy
    constexpr static int NumThreadsPerRow = Headdim > NThreads ? NThreads : Headdim;

    using GmemCopyTiledK = decltype(make_tiled_copy(
        Copy_Atom<DefaultCopy, Element>{},
        Layout<Shape<Int<NThreads / NumThreadsPerRow>, Int<NumThreadsPerRow>>,
               Stride<Int<NumThreadsPerRow>, _1>>{},
        Layout<Shape<_1, _1>>{}));
    using GmemCopyTiledV = GmemCopyTiledK;

    using MMA_ATOM = std::conditional_t<
        std::is_same_v<Element, cutlass::half_t>,
        cute::MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        cute::MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;

    using TiledMma = TiledMMA<
        MMA_ATOM,
        Layout<Shape<Int<NWarps>, _1, _1>>,
        Tile<Int<16 * NWarps>, _16, _16>>;
};

template <typename T, int Headdim_, int Deg_, int ExpandedDim_, int BlockD_, int BlockT_, int NWarps_, int OuterBlock_, int InnerBlock_, int PaddedExpandedDim_, bool DoubleBuffer_ = true>
struct Query_state_traits
{
    using Element = T;
    using index_t = int64_t;
    using ElementAccum = float;
    using C_type = int;
    #if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 750
        using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
        using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, Element>;
    #else
        using SmemCopyAtom = Copy_Atom<DefaultCopy, Element>;
        using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, Element>;
    #endif
    using multiindex_t = int32_t;

    // Number of threads
    static constexpr int NWarps = NWarps_;
    static constexpr int NThreads = NWarps * cutlass::NumThreadsPerWarp;
    constexpr static int Headdim = Headdim_;
    constexpr static int BlockD = BlockD_;
    constexpr static int BlockQ = BlockT_;
    constexpr static int BlockT = BlockT_;
    constexpr static int Deg = Deg_;
    constexpr static int ExpandedDim = ExpandedDim_;
    constexpr static int OuterBlock = OuterBlock_;
    constexpr static int InnerBlock = InnerBlock_;
    constexpr static int PaddedExpandedDim = PaddedExpandedDim_;
    constexpr static bool DoubleBuffer = DoubleBuffer_;
    // ================================ MMA ================================
    using MMA_ATOM = std::conditional_t<
        std::is_same_v<Element, cutlass::half_t>,
        cute::MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        cute::MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;

    using TiledMma = TiledMMA<
        MMA_ATOM,
        Layout<Shape<Int<NWarps>, _1, _1>>,
        Tile<Int<16 * NWarps>, _16, _16>>;


    // ================================ Q ================================
    constexpr static int BlockDimSmem = Headdim % 64 == 0 ? 64 : 32;
    constexpr static int kSwizzleB = BlockDimSmem == 32 ? 2 : 3;
    using SmemLayoutAtomQ = decltype(composition(Swizzle<kSwizzleB, 3, 3>{},
                                                  Layout<Shape<_8, Int<BlockDimSmem>>,
                                                         Stride<Int<BlockDimSmem>, _1>>{}));

    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        make_shape(Int<BlockQ>{}, Int<Headdim>{})));

    constexpr static int GmemThreadsPerRowQ = BlockDimSmem / 8;
    using GmemCopyTileQ = decltype(make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>{},
        Layout<Shape<Int<NThreads / GmemThreadsPerRowQ>, Int<GmemThreadsPerRowQ>>,
               Stride<Int<GmemThreadsPerRowQ>, _1>>{},
        Layout<Shape<_1, _8>>{}));

    // ================================ S ================================
    using SmemLayoutAtomS = decltype(composition(Swizzle<kSwizzleB, 3, 3>{},
                                                  Layout<Shape<_8, Int<BlockDimSmem>>,
                                                         Stride<Int<BlockDimSmem>, _1>>{}));
    using SmemLayoutS = decltype(tile_to_shape(
        SmemLayoutAtomS{},
        make_shape(Int<BlockD>{}, Int<Headdim>{})));
    using SmemLayoutSt = decltype(composition(SmemLayoutS{}, make_layout(Shape<Int<Headdim>, Int<BlockD>>{},
                                                                         GenRowMajor{})));
    using SmemLayoutStNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutSt{}));

    constexpr static int GmemThreadsPerRowS = BlockDimSmem / 8;
    using GmemCopyTileS = decltype(make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>{},
        Layout<Shape<Int<NThreads / GmemThreadsPerRowS>, Int<GmemThreadsPerRowS>>,
               Stride<Int<GmemThreadsPerRowS>, _1>>{},
        Layout<Shape<_1, _8>>{}));

    // ================================ N ================================
    using SmemLayoutN = Layout<Shape<Int<BlockD>>, Stride<_1>>;
    using GmemCopyTileN = decltype(make_tiled_copy(
        Copy_Atom<DefaultCopy, ElementAccum>{},
        Layout<Shape<Int<NThreads>>, Stride<_1>>{},
        Layout<Shape<_1>>{}));
    
    // ================================ O ================================
    using SmemLayoutAtomO = decltype(composition(Swizzle<kSwizzleB, 3, 3>{},
                                              Layout<Shape<_8, Int<BlockDimSmem>>,
                                                     Stride<Int<BlockDimSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<BlockT>, Int<Headdim>>{}));

    using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static constexpr int kGmemThreadsPerRow = Headdim / kGmemElemsPerLoad;
    using GmemLayoutAtom = Layout<Shape <Int<NThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemCopyTileO = decltype(make_tiled_copy(
        Copy_Atom<AutoVectorizingCopy, Element>{},
        GmemLayoutAtom{},
        Layout<Shape<_1, _8>>{})); // Tile Size: (NThreads / Headdim) x Headdim


    // ================================ PhiQ ================================
    using SmemLayoutPhiQ = Layout<Shape<Int<BlockQ>, Int<BlockD>>, Stride<Int<BlockD>, _1>>;

    // we don't care the performance on this one
    using GmemCopyTilePhiQ = decltype(make_tiled_copy(
        Copy_Atom<DefaultCopy, Element>{},
        Layout<Shape<Int<NThreads / BlockD>, Int<BlockD>>,
               Stride<Int<BlockD>, _1>>{},
        Layout<Shape<_1, _1>>{}));

    // ================================ NO ================================
    using SmemLayoutNO = Layout<Shape<Int<BlockQ>>, Stride<_1>>;
    using GmemCopyTileNO = GmemCopyTileN;

    // ================================ Y ================================
    using SmemLayoutY = SmemLayoutO;
    constexpr static int GmemThreadsPerRowY = BlockDimSmem / 8;;
    using GmemCopyTileY = decltype(make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>{},
        Layout<Shape<Int<NThreads / GmemThreadsPerRowY>, Int<GmemThreadsPerRowY>>,
               Stride<Int<GmemThreadsPerRowY>, _1>>{},
        Layout<Shape<_1, _8>>{}));
    
    // ================================ y ================================
    using SmemLayouty = SmemLayoutNO;
    using GmemCopyTiley = GmemCopyTileNO;

    // ================================ rowmax ================================
    using SmemLayoutRowmax = Layout<Shape<Int<BlockQ>>, Stride<_1>>;
    using GmemCopyTileRowmax = decltype(make_tiled_copy(
        Copy_Atom<DefaultCopy, float>{},
        Layout<Shape<Int<NThreads>>>{},
        Layout<Shape<_1>>{}));

    // ================================ log_G ================================
    using SmemLayoutLogG = SmemLayoutNO;
    using GmemCopyTileLogG = GmemCopyTileNO;

    // ================================ Smem Size ================================
    // smem size to store Q
    static constexpr int QSmemSize = size(SmemLayoutQ{}) * sizeof(Element);
    // smem size to store S
    static constexpr int SSmemSize = (DoubleBuffer ? 2 : 1) * size(SmemLayoutS{}) * sizeof(Element);
    // smem size to store norm
    static constexpr int NormSmemSize = (DoubleBuffer ? 2 : 1) * size(SmemLayoutN{}) * sizeof(ElementAccum);
    // smem size to store rowmax
    static constexpr int RowmaxSmemSize = size(SmemLayoutRowmax{}) * sizeof(float);
    // smem size to store Phi(Q)
    static constexpr int PhiQSmemSize = size(SmemLayoutPhiQ{}) * sizeof(Element);
    // smem size to store output
    static constexpr int OSmemSize = size(SmemLayoutO{}) * sizeof(Element);
    // smem size to store norm buffer
    static constexpr int NOSmemSize = size(SmemLayoutNO{}) * sizeof(ElementAccum);
    static constexpr int LogGSmemSize = size(SmemLayoutLogG{}) * sizeof(ElementAccum);
    // input smem size
    static constexpr int InputSmemSize = QSmemSize + SSmemSize + LogGSmemSize;
    // output smem size
    static constexpr int OutputSmemSize = OSmemSize;
};


template <typename T, int Headdim_, int Deg_, int ExpandedDim_, int BlockD_, int BlockT_, int NWarps_, int OuterBlock_, int InnerBlock_, int PaddedExpandedDim_, bool DoubleBuffer_, bool S_in_regs_ = false, typename Base = Query_state_traits<T, Headdim_, Deg_, ExpandedDim_, BlockD_, BlockT_, NWarps_, OuterBlock_, InnerBlock_, PaddedExpandedDim_, DoubleBuffer_>>
struct Query_state_bwd_traits : public Base
{
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    using SmemLayoutPhiQ = typename Base::SmemLayoutPhiQ;
    using multiindex_t = typename Base::multiindex_t;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;
    using SmemLayoutAtomQ = typename Base::SmemLayoutAtomQ;
    using MMA_ATOM = typename Base::MMA_ATOM;
    using SmemLayoutQ = typename Base::SmemLayoutQ;
    using SmemLayoutLogG = typename Base::SmemLayoutLogG;
    static_assert(std::is_same<Element, T>::value, "Element type must be the same as T");
    constexpr static int NWarps = Base::NWarps;
    constexpr static int NThreads = Base::NThreads;
    constexpr static int Headdim = Base::Headdim;
    constexpr static int BlockD = Base::BlockD;
    constexpr static int BlockQ = Base::BlockQ;
    constexpr static int BlockDimSmem = Base::BlockDimSmem;
    constexpr static int kSwizzleB = Base::kSwizzleB;
    constexpr static bool DoubleBuffer = Base::DoubleBuffer;
    constexpr static bool S_in_regs = S_in_regs_;
    static_assert(NThreads >= BlockD, "NThreads must be greater than or equal to BlockD");

    using SmemLayoutPhiQt = decltype(composition(SmemLayoutPhiQ{},
                                                 make_layout(
                                                     Shape<Int<BlockD>, Int<BlockQ>>{},
                                                     GenRowMajor{})));

    using SmemLayoutAtomdQ = decltype(composition(Swizzle<kSwizzleB, 3, 3>{},
                                                  Layout<Shape<_8, Int<BlockDimSmem>>,
                                                         Stride<Int<BlockDimSmem>, _1>>{}));
    using SmemLayoutdQ = decltype(tile_to_shape(
        SmemLayoutAtomdQ{},
        make_shape(Int<BlockQ>{}, Int<Headdim>{})));
    using SmemLayoutQt = decltype(composition(SmemLayoutdQ{}, make_layout(Shape<Int<Headdim>, Int<BlockQ>>{},
                                                                           GenRowMajor{})));
        
    using SmemCopyTiledQ = decltype(make_tiled_copy(
        Copy_Atom<DefaultCopy, Element>{},
        Layout<Shape<Int<NThreads>, _1>, Stride<_1, _1>>{}));
    using SmemCopyAtomdQ = Copy_Atom<DefaultCopy, Element>;

    using SmemLayoutdY = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        make_shape(Int<BlockQ>{}, Int<Headdim>{})));
    using SmemLayoutdYt = decltype(composition(SmemLayoutdY{},
                                               make_layout(Shape<Int<Headdim>, Int<BlockQ>>{},
                                                           GenRowMajor{})));
    using SmemLayoutdy = Layout<Shape<Int<BlockQ>>>;
    using SmemLayoutdS = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        make_shape(Int<BlockD>{}, Int<Headdim>{})));
    using SmemLayoutdN = Layout<Shape<Int<BlockD>>, Stride<_1>>;
    using SmemLayoutdNBuffer = Layout<Shape<Int<NWarps>>, Stride<_1>>;
    using SmemLayoutdQaccum = Layout<Shape<Int<BlockQ>, Int<BlockD>>, Stride<Int<BlockD>, _1>>;
    using SmemLayoutRowmax = Layout<Shape<Int<BlockQ>>>;

    constexpr static int QSmemSize = size(SmemLayoutQ{}) * sizeof(Element);
    constexpr static int PhiQSmemSize = size(SmemLayoutPhiQ{}) * sizeof(ElementAccum);
    constexpr static int SSmemSize = size(SmemLayoutdS{}) * sizeof(Element);
    constexpr static int NSmemSize = size(SmemLayoutdN{}) * sizeof(ElementAccum);
    constexpr static int dYSmemSize = size(SmemLayoutdY{}) * sizeof(Element);
    constexpr static int dySmemSize = size(SmemLayoutdy{}) * sizeof(ElementAccum);
    constexpr static int dNSmemSize = size(SmemLayoutdN{}) * sizeof(ElementAccum);
    constexpr static int dQaccumSmemSize = size(SmemLayoutdQaccum{}) * sizeof(ElementAccum);
    constexpr static int dNBufferSmemSize = size(SmemLayoutdNBuffer{}) * sizeof(ElementAccum);
    constexpr static int LogGSmemSize = size(SmemLayoutLogG{}) * sizeof(ElementAccum);
    constexpr static int RowmaxSmemSize = size(SmemLayoutRowmax{}) * sizeof(float);

    constexpr static int InputSmemSizedSdN = (DoubleBuffer ? 2 : 1) * (QSmemSize + dYSmemSize + LogGSmemSize);
    constexpr static int OutputSmemSizedSdN = SSmemSize;

    constexpr static int InputSmemSizedQ = QSmemSize + LogGSmemSize + (DoubleBuffer ? 2 : 1) * (SSmemSize) + dYSmemSize;
    constexpr static int OutputSmemSizedQ = QSmemSize + LogGSmemSize;

    constexpr static int SmemSizedSdNdQ = (DoubleBuffer ? 2 : 1) * (QSmemSize + LogGSmemSize) + dYSmemSize + SSmemSize + dQaccumSmemSize;

    constexpr static int Deg = Deg_;

    static_assert(Deg == 2 || Deg == 4, "Deg must be 2 or 4");

    using GmemCopyTileQ = typename Base::GmemCopyTileQ;
    using GmemCopyTileS = typename Base::GmemCopyTileS;
    using GmemCopyTileN = typename Base::GmemCopyTileN;
    using GmemCopyTiledY = typename Base::GmemCopyTileQ;
    using GmemCopyTiledLogG = typename Base::GmemCopyTileLogG;
    using GmemCopyTileRowmax = typename Base::GmemCopyTileRowmax;

    constexpr static int NumThreadsPerRow = Headdim > NThreads ? NThreads : Headdim;

    using GmemCopyTiledQ = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<uint16_t>, Element>{},
        Layout<Shape<Int<NThreads / NumThreadsPerRow>, Int<NumThreadsPerRow>>,
               Stride<Int<NumThreadsPerRow>, _1>>{},
        Layout<Shape<_1, _1>>{}));

    using GmemCopyTiledS = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<uint16_t>, Element>{},
        Layout<Shape<Int<NThreads / NumThreadsPerRow>, Int<NumThreadsPerRow>>,
               Stride<Int<NumThreadsPerRow>, _1>>{},
        Layout<Shape<_1, _1>>{}));

    using GmemCopyTiledy = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<uint32_t>, ElementAccum>{},
        Layout<Shape<Int<NThreads>>, Stride<_1>>{},
        Layout<Shape<_1>>{}));

    using AtomShape_MNK = typename MMA_ATOM::Shape_MNK;
    constexpr static int AtomShape_M = size<0>(AtomShape_MNK{});
    constexpr static int AtomShape_N = size<1>(AtomShape_MNK{});
    constexpr static int dSWarpN = (Headdim / AtomShape_N) > NWarps ? NWarps : (Headdim / AtomShape_N);
    constexpr static int dSWarpM = NWarps / dSWarpN;

    using TiledMma = typename Base::TiledMma;
};