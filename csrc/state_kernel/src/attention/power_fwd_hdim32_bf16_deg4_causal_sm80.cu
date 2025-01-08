// Copyright (c) 2024, Sean Zhang.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "power_fwd_launch_template.h"

template<>
void run_mha_fwd_<cutlass::bfloat16_t, 32, 4, true>(Power_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim32<cutlass::bfloat16_t, 4, true>(params, stream);
}
