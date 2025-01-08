// Copyright (c) 2024, Sean Zhang.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "power_bwd_launch_template.h"

template<>
void run_mha_bwd_<cutlass::bfloat16_t, 64, 3>(Power_bwd_params &params, cudaStream_t stream) {
    run_mha_bwd_hdim64<cutlass::bfloat16_t, 3>(params, stream);
}
