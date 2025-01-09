from functools import lru_cache

import torch
import torch.utils.benchmark as benchmark
from power_attention_cuda import (
    expand_dim,
    expanded_key_value_product,
    expanded_query_state_product,
    multi_index,
)


@lru_cache(maxsize=10)
def multi_indices_(d, p):
    return multi_index(d, p, True)  # noqa: FBT003


def multi_indices(d, p, padded=True):
    indices, coeff = multi_indices_(d, p)
    D = expand_dim(d, p)
    if padded:
        return indices, coeff
    return indices[:D, :], coeff[:D]


def benchmark_forward(
    fn,
    *inputs,
    repeats=10,
    desc='',
    verbose=False,
    amp=False,
    amp_dtype=torch.float16,
    **kwinputs,
):
    """Use Pytorch Benchmark on the forward pass of an arbitrary function."""
    if verbose:
        print(desc, '- Forward pass')

    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)

    t = benchmark.Timer(
        stmt='fn_amp(*inputs, **kwinputs)',
        globals={'fn_amp': amp_wrapper, 'inputs': inputs, 'kwinputs': kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def run_expanded_key_value_product():
    print('Expanded Key Value Product')
    b, n, c, h, d = 1, 2, 1024, 12, 32
    p = 4
    dtype = torch.bfloat16
    k = torch.randn(b, n, c, h, d, device='cuda', dtype=dtype)
    v = torch.randn(b, n, c, h, d, device='cuda', dtype=dtype)
    indices, coeffs = multi_indices(d, p)  # get multiindices
    D = expand_dim(d, p)  # get expanded dimension

    def ekv(k, v):
        return expanded_key_value_product(
            k,
            v,
            indices,
            coeffs,
            p,
            False,  # whether to return expanded vector(key)
        )

    t, m = benchmark_forward(ekv, k, v, verbose=True)


def run_expanded_vector_state_product():
    print('Expanded Vector State Product')
    b, n, c, h, d = 1, 2, 1024, 12, 32
    p = 4
    dtype = torch.bfloat16
    x = torch.randn(b, n, c, h, d, device='cuda', dtype=dtype)
    indices, coeffs = multi_indices(d, p)  # get multiindices
    D = expand_dim(d, p)  # get expanded dimension
    expanded_D = len(coeffs)
    stabilizer = (
        1.0 if dtype == torch.bfloat16 else float(D)
    )  # stabilizer is only needed for float16
    s = torch.randn(b, n, h, expanded_D, d, device='cuda', dtype=dtype)

    def evs(x):
        return expanded_query_state_product(
            x,
            s,
            indices,
            coeffs,
            p,
            stabilizer,
            False,  # whether to return expanded vector(query)
        )

    t, m = benchmark_forward(evs, x, verbose=True)


if __name__ == '__main__':
    run_expanded_key_value_product()
    run_expanded_vector_state_product()
