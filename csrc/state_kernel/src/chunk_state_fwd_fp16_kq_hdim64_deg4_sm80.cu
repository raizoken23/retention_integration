// Copyright (c) 2024. Sean Zhang.
// Splitting the different degrees to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "chunk_state_launch_template.h"

template<>
void run_compute_chunk_states<cutlass::half_t, 64, 4>(Chunk_state_params &params, cudaStream_t stream) {
    run_chunk_states_fwd_<cutlass::half_t, 64, 4>(params, stream);
}
