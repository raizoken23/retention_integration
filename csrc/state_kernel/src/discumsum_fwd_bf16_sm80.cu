// Copyright (c) 2024. Sean Zhang.
// Splitting the different degrees to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "discumsum_launch_template.h"

template<>
void run_discumsum_fwd<cutlass::bfloat16_t>(Discumsum_params &params, cudaStream_t stream) {
    run_discumsum_fwd_<cutlass::bfloat16_t>(params, stream);
}
