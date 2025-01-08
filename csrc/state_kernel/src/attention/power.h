/******************************************************************************
 * Copyright (c) 2024, Sean Zhang.
 ******************************************************************************/

#pragma once

#include <vector>

constexpr int TOTAL_DIM = 0;
constexpr int H_DIM = 1;
constexpr int D_DIM = 2;

////////////////////////////////////////////////////////////////////////////////////////////////////

enum SimilarityType {
    sympower,
    softmax
};

struct Qkv_params {
    using index_t = int64_t;
    // The QKV matrices.
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;

    // The number of heads.
    int h, h_k;
    // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
    // different from nheads (query).
    int h_h_k_ratio; // precompute h / h_k,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Power_fwd_params : public Qkv_params {
    // The gating matrix (cumulative log)
    void *__restrict__ log_g_q_ptr;
    void *__restrict__ log_g_k_ptr;

    // The O matrix (output).
    void * __restrict__ o_ptr;
    void * __restrict__ oaccum_ptr;

    // The stride between rows of O.
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    // The stride between rows of norm.
    index_t norm_batch_stride;
    index_t norm_row_stride;
    index_t norm_head_stride;
    index_t rowmax_batch_stride;
    index_t rowmax_row_stride;
    index_t rowmax_head_stride;

    // The pointer to the P matrix.
    void * __restrict__ p_ptr;

    // The pointer to the norm vector.
    void * __restrict__ norm_ptr;

    // The pointer to the rowmax vector.
    void * __restrict__ rowmax_ptr;

    // The dimensions.
    int b, seqlen_q, seqlen_k, seqlen_knew, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded, rotary_dim, total_q;

    // The scaling factors for the kernel.
    float scale_softmax_log2;

    // array of length b+1 holding starting offset of each sequence.
    int * __restrict__ cu_seqlens_q;
    int * __restrict__ cu_seqlens_k;
    int * __restrict__ leftpad_k;

    // If provided, the actual length of each k sequence.
    int * __restrict__ seqused_k;

    int *__restrict__ blockmask;

    // The K_new and V_new matrices.
    void * __restrict__ knew_ptr;
    void * __restrict__ vnew_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t knew_batch_stride;
    index_t vnew_batch_stride;
    index_t knew_row_stride;
    index_t vnew_row_stride;
    index_t knew_head_stride;
    index_t vnew_head_stride;
    index_t g_row_stride;
    index_t g_batch_stride;
    index_t g_head_stride;

    // The cos and sin matrices for rotary embedding.
    void * __restrict__ rotary_cos_ptr;
    void * __restrict__ rotary_sin_ptr;

    // The indices to index into the KV cache.
    int * __restrict__ cache_batch_idx;

    // Paged KV cache
    int * __restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;

    // Local window size
    int window_size_left, window_size_right;

    // Stablizer for the norm
    float stabilizer;
    float log_stabilizer;
    float stabilizer_p;
    bool use_multiplier;

    // ε
    float ε;

    bool is_bf16;
    bool is_causal;
    bool flash_equivalent;

    // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
    // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
    bool is_seqlens_k_cumulative;

    bool is_rotary_interleaved;

    int num_splits;  // For split-KV version

    int deg;

    bool normal_space;
    
    friend std::ostream &operator<<(std::ostream &os, const Power_fwd_params &params) {
        os << "deg: " << params.deg << "\n";
        os << "window_size_left: " << params.window_size_left << "\n";
        os << "window_size_right: " << params.window_size_right << "\n";
        os << "q_batch_stride: " << params.q_batch_stride << "\n";
        os << "k_batch_stride: " << params.k_batch_stride << "\n";
        os << "v_batch_stride: " << params.v_batch_stride << "\n";
        os << "q_row_stride: " << params.q_row_stride << "\n";
        os << "k_row_stride: " << params.k_row_stride << "\n";
        os << "v_row_stride: " << params.v_row_stride << "\n";
        os << "q_head_stride: " << params.q_head_stride << "\n";
        os << "k_head_stride: " << params.k_head_stride << "\n";
        os << "v_head_stride: " << params.v_head_stride << "\n";
        os << "cu_seqlens_q: " << params.cu_seqlens_q << "\n";
        os << "cu_seqlens_k: " << params.cu_seqlens_k << "\n";
        os << "seqused_k: " << params.seqused_k << "\n";
        os << "block_table: " << params.block_table << "\n";
        os << "block_table_batch_stride: " << params.block_table_batch_stride << "\n";
        os << "page_block_size: " << params.page_block_size << "\n";
        os << "rotary_cos_ptr: " << params.rotary_cos_ptr << "\n";
        os << "rotary_sin_ptr: " << params.rotary_sin_ptr << "\n";
        os << "cache_batch_idx: " << params.cache_batch_idx << "\n";
        os << "p_ptr: " << params.p_ptr << "\n";
        os << "norm_ptr: " << params.norm_ptr << "\n";
        os << "rowmax_ptr: " << params.rowmax_ptr << "\n";
        os << "is_bf16: " << params.is_bf16 << "\n";
        os << "is_causal: " << params.is_causal << "\n";
        os << "is_seqlens_k_cumulative: " << params.is_seqlens_k_cumulative << "\n";
        os << "is_rotary_interleaved: " << params.is_rotary_interleaved << "\n";
        os << "num_splits: " << params.num_splits << "\n";
        os << "ε: " << params.ε << "\n";
        os << "stabilizer: " << params.stabilizer << "\n";
        os << "log_stabilizer: " << params.log_stabilizer << "\n";
        return os;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Power_bwd_params : public Power_fwd_params {

    // The dY, dy and dQKV matrices.
    void *__restrict__ dY_ptr;
    void *__restrict__ dy_ptr;
    void *__restrict__ dq_ptr;
    void *__restrict__ dk_ptr;
    void *__restrict__ dv_ptr;
    void *__restrict__ dq_accum_ptr;
    void *__restrict__ rowmax_ptr;

    // dlog_g_q and dlog_g_k
    void *__restrict__ dlog_g_q_ptr;
    void *__restrict__ dlog_g_k_ptr;

    // // To accumulate dK and dV in case we're splitting the bwd along seqlen_q
    // dimension void *__restrict__ dk_accum_ptr; void *__restrict__
    // dv_accum_ptr;

    // The stride between rows of the dO, dQ, dK and dV matrices.
    // TD [2022-04-16]: We're using 32-bit indexing to save registers.
    // The code probably won't work for arrays larger than 2GB.
    index_t dY_batch_stride;
    index_t dY_row_stride;
    index_t dY_head_stride;
    index_t dy_batch_stride;
    index_t dy_row_stride;
    index_t dy_head_stride;
    index_t dq_batch_stride;
    index_t dk_batch_stride;
    index_t dv_batch_stride;
    index_t dq_row_stride;
    index_t dk_row_stride;
    index_t dv_row_stride;
    index_t dq_head_stride;
    index_t dk_head_stride;
    index_t dv_head_stride;
    index_t dq_accum_split_stride;
    index_t rowmax_batch_stride;
    index_t rowmax_row_stride;
    index_t rowmax_head_stride;
    index_t dg_batch_stride;
    index_t dg_row_stride;
    index_t dg_head_stride;

    int deg;
    bool deterministic;

    friend std::ostream &operator<<(std::ostream &os, const Power_bwd_params &params) {
        os << "Power_bwd_params\n";
        os << "deg: " << params.deg << "\n";
        os << "deterministic: " << params.deterministic << "\n";
        os << "window_size_left: " << params.window_size_left << "\n";
        os << "window_size_right: " << params.window_size_right << "\n";
        os << "dY_batch_stride: " << params.dY_batch_stride << "\n";
        os << "dY_row_stride: " << params.dY_row_stride << "\n";
        os << "dY_head_stride: " << params.dY_head_stride << "\n";
        os << "dy_batch_stride: " << params.dy_batch_stride << "\n";
        os << "dy_row_stride: " << params.dy_row_stride << "\n";
        os << "dy_head_stride: " << params.dy_head_stride << "\n";
        os << "dq_batch_stride: " << params.dq_batch_stride << "\n";
        os << "dq_row_stride: " << params.dq_row_stride << "\n";
        os << "dq_head_stride: " << params.dq_head_stride << "\n";
        os << "dk_batch_stride: " << params.dk_batch_stride << "\n";
        os << "dv_batch_stride: " << params.dv_batch_stride << "\n";
        os << "dk_row_stride: " << params.dk_row_stride << "\n";
        os << "dv_row_stride: " << params.dv_row_stride << "\n";
        os << "dk_head_stride: " << params.dk_head_stride << "\n";
        os << "dv_head_stride: " << params.dv_head_stride << "\n";
        os << "dq_accum_split_stride: " << params.dq_accum_split_stride << "\n";
        os << "rowmax_batch_stride: " << params.rowmax_batch_stride << "\n";
        os << "rowmax_row_stride: " << params.rowmax_row_stride << "\n";
        os << "rowmax_head_stride: " << params.rowmax_head_stride << "\n";
        return os;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int Headdim, int DEG, bool Is_causal> void run_mha_fwd_(Power_fwd_params &params, cudaStream_t stream);
template<typename T, int Headdim, int DEG, bool Is_causal> void run_mha_fwd_splitkv_dispatch(Power_fwd_params &params, cudaStream_t stream);

template<typename T, int Headdim, int DEG> void run_mha_bwd_(Power_bwd_params &params, cudaStream_t stream);
