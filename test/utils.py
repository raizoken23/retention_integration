import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from einops import rearrange
from power_attention.power_full import PowerAttentionKernel
# from power_attention.power_attention import SymmetricStateKernel
from torch.utils._pytree import tree_map

def configs(shape, chunk_sizes, dtypes, degs, εs, ref_dtype=None):
    return [
        (shape, chunk, p, dtype, ε, ref_dtype)
        for dtype in dtypes
        for p in degs
        for chunk in chunk_sizes
        for ε in εs
    ]


def paramify(xs):
    return tree_map(lambda x: None if x is None else torch.nn.Parameter(x.detach().clone().requires_grad_(x.requires_grad)), xs)

def pytree_to(xs, dtype):
    return tree_map(lambda x: None if x is None else x.to(dtype=dtype), xs)

def diff_lower_than(a, b, atol=1e-3):
    return ((a - b).abs() < atol).float().mean()


cfg_1_128_1_32__128 = configs([1, 256, 1, 32], [128], [torch.bfloat16, torch.float16], [2], [1e-6], torch.float32)
cfg_short_head64 = configs([1, 192, 1, 64], [64], [torch.float16, torch.bfloat16], [2], [1e-5], torch.float32)
cfg_short_deg4 = configs([1, 256, 1, 64], [128], [torch.float16, torch.bfloat16], [4], [1e-5], torch.float32)
cfg_full_deg2 = configs([1, 3072, 2, 32], [1024], [torch.float16, torch.bfloat16], [2], [1e-5], torch.float32)
cfg_full_head64 = configs([1, 3072, 2, 64], [1024], [torch.float16, torch.bfloat16], [2], [1e-5], torch.float32)
cfg_full = configs([1, 3072, 2, 32], [1024], [torch.float16, torch.bfloat16], [2, 4], [1e-5], torch.float32)
cfg_full_64 = configs([1, 3072, 2, 64], [1024], [torch.float16, torch.bfloat16], [2], [1e-5], torch.float32)
cfg_more = configs([1, 4096, 1, 32], [64], [torch.bfloat16], [2, 4], [1e-5], torch.float32)
cfg_long = configs([1, 8192, 12, 64], [4096], [torch.bfloat16], [4], [1e-5], torch.float32)


def check(a, b, atol, rtol, violation_proportion=1e-3, verbose=False, rtol_mean=None):
    diff = torch.abs(a - b)
    rel_diff = diff / torch.abs(b)
    mean_rel_diff = torch.nan_to_num(rel_diff, 0, 0, 0).mean()
    median_rel_diff = torch.nan_to_num(rel_diff, 0, 0, 0).median()
    violations = (diff > atol + rtol * torch.abs(b)).sum()
    fail = False
    if rtol_mean is not None:
        fail = mean_rel_diff > rtol_mean
    elif violations > violation_proportion * diff.numel():
        fail = True

    if fail:
        if verbose:
            print(f'Shape(a): {a.shape}')
            print(f'Shape(b): {b.shape}')
            print(f'Max diff: {diff.max()}')
            print(f'Max rel diff: {torch.nan_to_num(rel_diff, 0, 0, 0).max()}')
            print(f'Mean diff: {diff.mean()}')
            print(f'Mean rel diff: {mean_rel_diff}')
            print(f'Std diff: {diff.std()}')
            print(f'Mean(abs(a)): {a.abs().mean()}')
            print(f'Mean(abs(b)): {b.abs().mean()}')
            print(f'a: {a}')
            print(f'b: {b}')
            print(f'diff: {diff}')
            print(f'rel_diff: {rel_diff}')
            print(f'a norm: {a.norm()}')
            print(f'b norm: {b.norm()}')
        msg = f'Too many ({100 * violations / diff.numel():.3f}% > {100 * violation_proportion:.3f}%) violations, mean relative diff is {mean_rel_diff}, median relative diff is {median_rel_diff}, tensors not matching'
        raise AssertionError(msg)


def compare(ref_val, cuda_val, torch_val, allowance=3, precision=1e-3, abs_tol=None, verbose_name=None):
    cuda_ref_diff = torch.abs(ref_val - cuda_val)
    torch_ref_diff = torch.abs(ref_val - torch_val)
    cuda_precision = (cuda_ref_diff < precision).float().mean()
    torch_precision = (torch_ref_diff < precision).float().mean()
    if verbose_name:
        print(f'Cuda {verbose_name} precision: {cuda_precision}')
        print(f'Torch {verbose_name} precision: {torch_precision}')
    if abs_tol is not None:
        assert cuda_ref_diff.max() <= torch_ref_diff.max() * allowance or cuda_ref_diff.max() <= abs_tol or cuda_precision >= torch_precision
    else:
        assert cuda_ref_diff.max() <= torch_ref_diff.max() * allowance
    


def create_QKVR(b, t, h, d, dtype, gating=False, log_gating=True, chunk_size=None, device='cuda'):
    """Create random Q, K, V tensors, optionally with gating coefficients"""
    Q = (
        torch.rand((b, t, h, d), dtype=dtype)
        .to(device)
    )
    K = (
        torch.rand((b, t, h, d), dtype=dtype)
        .to(device)
    )
    V = (
        torch.rand((b, t, h, d), dtype=dtype)
        .to(device)
    )
    if chunk_size is not None:
        Q, K, V = tree_map(
            lambda X: rearrange(
                X, 'b (n c) h d -> b n c h d', c=chunk_size
            ),
            (Q, K, V),
        )
    if gating:
        R = torch.rand([b, t, h], dtype=torch.float32).to(device) * 10
        if log_gating:
            log_Γ = F.logsigmoid(R.detach()).cumsum(1)
            if chunk_size is not None:
                log_G = rearrange(log_Γ, 'b (n c) h -> b n c h', c=chunk_size)
            else:
                log_G = log_Γ
            R = log_G
    else:
        R = None
    return Q, K, V, R

def create_SN(b, n, h, d, D, dtype):
    """Create random state and state norm. n should equal t // c"""
    S = torch.rand([b, n, h, D, d], dtype=dtype).to('cuda')
    s = torch.rand([b, n, h, D], dtype=torch.float32).to('cuda')
    return S, s

def to_csv(tensor, filename=None):
    """
    Save a tensor to a CSV file.
    """
    if filename is None:
        filename = f'tensor_{tensor.shape}.csv'
    t = tensor.squeeze().detach().cpu().numpy()

    import pandas as pd

    df = pd.DataFrame(t)
    df.to_csv(filename, index=False, header=False)
    print(f"Tensor saved to {filename}")
