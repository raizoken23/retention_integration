# Copied from Driss Guessous's PR in PyTorch: https://github.com/pytorch/pytorch/pull/105602

# This file is run to generate the kernel instantiations for the flash_attn kernels
# They are written to several files in order to speed up compilation

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

DTYPE_MAP = {
    "fp16": "cutlass::half_t",
    "bf16": "cutlass::bfloat16_t",
}

SM = [80]  # Sm80 kernels support up to
HEAD_DIMENSIONS = [32, 64]
DEGREES = [1, 2, 3, 4]  # Added degrees
IS_CAUSAL = ["true"]
KERNEL_IMPL_TEMPLATE_FWD = """#include "power_fwd_launch_template.h"

template<>
void run_mha_fwd_<{DTYPE}, {HEAD_DIM}, {DEG}, {IS_CAUSAL}>(Power_fwd_params &params, cudaStream_t stream) {{
    run_mha_fwd_hdim{HEAD_DIM}<{DTYPE}, {DEG}, {IS_CAUSAL}>(params, stream);
}}
"""

KERNEL_IMPL_TEMPLATE_FWD_SPLIT = """#include "power_fwd_launch_template.h"

template void run_mha_fwd_splitkv_dispatch<{DTYPE}, {HEAD_DIM}, {IS_CAUSAL}>(Power_fwd_params &params, cudaStream_t stream);
"""

KERNEL_IMPL_TEMPLATE_BWD = """#include "power_bwd_launch_template.h"

template<>
void run_mha_bwd_<{DTYPE}, {HEAD_DIM}, {DEG}>(Power_bwd_params &params, cudaStream_t stream) {{
    run_mha_bwd_hdim{HEAD_DIM}<{DTYPE}, {DEG}>(params, stream);
}}
"""


@dataclass
class Kernel:
    sm: int
    dtype: str
    head_dim: int
    deg: int
    is_causal: bool
    direction: str

    @property
    def template(self) -> str:
        if self.direction == "fwd":
            return KERNEL_IMPL_TEMPLATE_FWD.format(
                DTYPE=DTYPE_MAP[self.dtype], 
                HEAD_DIM=self.head_dim,
                DEG=self.deg,
                IS_CAUSAL=self.is_causal
            )
        else:
            return KERNEL_IMPL_TEMPLATE_BWD.format(
                DTYPE=DTYPE_MAP[self.dtype], 
                HEAD_DIM=self.head_dim,
                DEG=self.deg
            )

    @property
    def filename(self) -> str:
        return f"power_{self.direction}_hdim{self.head_dim}_{self.dtype}_deg{self.deg}_{'causal_' if self.is_causal == 'true' else ''}sm{self.sm}.cu"


def get_all_kernels() -> List[Kernel]:
    for direction in ['fwd']:
        for dtype, head_dim, deg, is_causal, sm in itertools.product(
            DTYPE_MAP.keys(), HEAD_DIMENSIONS, DEGREES, IS_CAUSAL, SM
        ):
            yield Kernel(sm=sm, dtype=dtype, head_dim=head_dim, deg=deg, is_causal=is_causal, direction=direction)
    
    for direction in ["bwd"]:
        for dtype, head_dim, deg, sm in itertools.product(
            DTYPE_MAP.keys(), HEAD_DIMENSIONS, DEGREES, SM
        ):
            yield Kernel(sm=sm, dtype=dtype, head_dim=head_dim, deg=deg, is_causal="false", direction=direction)


def write_kernel(kernel: Kernel, autogen_dir: Path) -> None:
    prelude = """// Copyright (c) 2024, Sean Zhang.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"\n
"""
    (autogen_dir / kernel.filename).write_text(prelude + kernel.template)


def main(output_dir: Optional[str]) -> None:
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)

    for kernel in get_all_kernels():
        write_kernel(kernel, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate_kernels",
        description="Generate the power_attention kernels template instantiations",
    )
    # Set an optional output directory
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="Where to generate the kernels "
        " will default to the current directory ",
    )
    args = parser.parse_args()
    main(args.output_dir)
