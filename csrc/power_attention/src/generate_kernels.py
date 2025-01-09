# Copied from Tri Dao's work https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/generate_kernels.py.print
# Splitting the different head dimensions to different files to speed up compilation.


import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DTYPE_MAP = {
    'fp16': 'cutlass::half_t',
    'bf16': 'cutlass::bfloat16_t',
    'fp32': 'float',
}

SM = [80]  # Sm80 kernels support up to
DEG = [2]
HEAD_DIMENSIONS = [32, 64]
CHUNK_STATE_FWD = """#include "chunk_state_launch_template.h"

template<>
void run_compute_chunk_states<{DTYPE}, {KQ_HEAD_DIM}, {DEG}>(Chunk_state_params &params, cudaStream_t stream) {{
    run_chunk_states_fwd_<{DTYPE}, {KQ_HEAD_DIM}, {DEG}>(params, stream);
}}
"""
QUERY_STATE_FWD = """#include "query_state_launch_template.h"

template<>
void run_compute_query_states<{DTYPE}, {KQ_HEAD_DIM}, {DEG}>(Query_state_params &params, cudaStream_t stream) {{
    run_compute_query_states_<{DTYPE}, {KQ_HEAD_DIM}, {DEG}>(params, stream);
}}
"""
QUERY_STATE_BWD = """#include "query_state_launch_template.h"

template<>
void run_compute_query_states_bwd<{DTYPE}, {KQ_HEAD_DIM}, {DEG}>(Query_state_bwd_params &params, cudaStream_t stream) {{
    run_compute_query_states_bwd_<{DTYPE}, {KQ_HEAD_DIM}, {DEG}>(params, stream);
}}
"""
CHUNK_STATE_BWD = """#include "chunk_state_launch_template.h"

template<>
void run_compute_chunk_states_bwd<{DTYPE}, {KQ_HEAD_DIM}, {DEG}>(Chunk_state_bwd_params &params, cudaStream_t stream) {{
    run_compute_chunk_states_bwd_<{DTYPE}, {KQ_HEAD_DIM}, {DEG}>(params, stream);
}}
"""
DISCUMSUM_FWD = """#include "discumsum_launch_template.h"

template<>
void run_discumsum_fwd<{DTYPE}>(Discumsum_params &params, cudaStream_t stream) {{
    run_discumsum_fwd_<{DTYPE}>(params, stream);
}}
"""
DISCUMSUM_BWD = """#include "discumsum_launch_template.h"

template<>
void run_discumsum_bwd<{DTYPE}>(Discumsum_bwd_params &params, cudaStream_t stream) {{
    run_discumsum_bwd_<{DTYPE}>(params, stream);
}}
"""

KERNEL_TEMPLATE_MAP = {
    'chunk_state_fwd': CHUNK_STATE_FWD,
    'chunk_state_bwd': CHUNK_STATE_BWD,
    'query_state_fwd': QUERY_STATE_FWD,
    'query_state_bwd': QUERY_STATE_BWD,
}

KERNEL_ON_DTYPE_MAP = {
    'discumsum_fwd': DISCUMSUM_FWD,
    'discumsum_bwd': DISCUMSUM_BWD,
}


@dataclass
class Kernel:
    sm: int
    dtype: str
    kq_head_dim: int
    # v_head_dim: int
    deg: int
    part: str

    @property
    def template(self) -> str:
        return KERNEL_TEMPLATE_MAP[f'{self.part}'].format(
            DTYPE=DTYPE_MAP[self.dtype],
            KQ_HEAD_DIM=self.kq_head_dim,
            # V_HEAD_DIM=self.v_head_dim,
            DEG=self.deg,
        )

    @property
    def filename(self) -> str:
        return f'{self.part}_{self.dtype}_kq_hdim{self.kq_head_dim}_deg{self.deg}_sm{self.sm}.cu'
    

@dataclass
class KernelOnDtype:
    sm: int
    dtype: str
    part: str

    @property
    def template(self) -> str:
        return KERNEL_ON_DTYPE_MAP[f'{self.part}'].format(
            DTYPE=DTYPE_MAP[self.dtype],
        )
    
    @property
    def filename(self) -> str:
        return f'{self.part}_{self.dtype}_sm{self.sm}.cu'


def get_all_kernels() -> list[Kernel]:
    for part in KERNEL_TEMPLATE_MAP.keys():
        for deg in DEG:
            for dtype, kq_head_dim, sm in itertools.product(
                ['fp16', 'bf16'], HEAD_DIMENSIONS, SM
            ):
                yield Kernel(
                    part=part,
                    sm=sm,
                    dtype=dtype,
                    kq_head_dim=kq_head_dim,
                    # v_head_dim=v_head_dim,
                    deg=deg,
                )

    for part in KERNEL_ON_DTYPE_MAP.keys():
        for dtype, sm in itertools.product(
            DTYPE_MAP.keys(), SM
        ):
            yield KernelOnDtype(
                part=part,
                sm=sm,
                dtype=dtype,
            )


def write_kernel(kernel: Kernel, autogen_dir: Path) -> None:
    prelude = """// Copyright (c) 2024. Sean Zhang.
// Splitting the different degrees to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"\n
"""
    (autogen_dir / kernel.filename).write_text(prelude + kernel.template)
    print(f'Wrote {kernel.filename}')


def main(output_dir: Optional[str]) -> None:
    output_dir = Path(__file__).parent if output_dir is None else Path(output_dir)

    for kernel in get_all_kernels():
        write_kernel(kernel, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='generate_kernels',
        description='Generate the flash_attention kernels template instantiations',
    )
    # Set an optional output directory
    parser.add_argument(
        '-o',
        '--output_dir',
        required=False,
        help='Where to generate the kernels ' ' will default to the current directory ',
    )
    args = parser.parse_args()
    main(args.output_dir)
