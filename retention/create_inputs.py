import torch
from torch.utils._pytree import tree_map
import math
import torch.nn.functional as F
from vidrial.py_utils.common import default_d_tile
from vidrial.kernels.sympow_mma.dimensions import sympow_dim
from retention._utils import compute_expanded_dim

## Useful function to create sample inputs
def create_inputs(b=2, t=1024, h=8, d=32, qhead_ratio=1, dtype=torch.bfloat16, device='cuda', gating=False,
                  chunk_size=None, deg=2, requires_grad=False, seed=42, scale=1.0, switch_over_seq_len=None):
    torch.manual_seed(seed)
    Q = torch.randn(size=(b, t, h * qhead_ratio, d), dtype=dtype, device=device) / math.sqrt(d)
    K = torch.randn(size=(b, t, h, d), dtype=dtype, device=device) / math.sqrt(d)
    V = torch.randn(size=(b, t, h, d), dtype=dtype, device=device) / math.sqrt(d)
    if gating:
        log_G = F.logsigmoid(torch.randn(size=(b, t, h), dtype=torch.float32, device=device))
    else:
        log_G = None
    initial_state = None
    if requires_grad:
        Q, K, V, log_G, initial_state = tree_map(
            lambda x: x.clone().detach().requires_grad_(True) if x is not None else None, (Q, K, V, log_G, initial_state))
    return dict(Q=Q, K=K, V=V, log_G=log_G, 
                initial_state=initial_state,
                return_final_state=False,
                deg=deg, scale=scale,
                chunk_size=chunk_size,
                switch_over_seq_len=switch_over_seq_len)

def create_inference_inputs(b=2, tq=1, tk=1024, h=8, d=32, qhead_ratio=1, dtype=torch.bfloat16, device='cuda', gating=False,
                  chunk_size=None, deg=2, requires_grad=False, seed=42, scale=1.0, initial_state=False, switch_over_seq_len=512, fused_normalizer=True):
    torch.manual_seed(seed)
    assert tk > 0 or initial_state, 'tk must be greater than 0 or initial_state must be True'
    Q = torch.randn(size=(b, tq, h * qhead_ratio, d), dtype=dtype, device=device) / d**0.5
    K = torch.randn(size=(b, tk, h, d), dtype=dtype, device=device) / d**0.5 if tk > 0 else None
    V = torch.randn(size=(b, tk, h, d), dtype=dtype, device=device) / d**0.5 if tk > 0 else None
    if gating:
        log_G = F.logsigmoid(6.906768 + torch.randn(size=(b, tk, h), dtype=torch.float32, device=device)) if tk > 0 else None
    else:
        log_G = None
    D = sympow_dim(d, deg, d_tile=default_d_tile(d, deg)) if fused_normalizer else compute_expanded_dim(d, deg=deg)
    initial_state = None if not initial_state else torch.randn(size=(b, h, D, d), dtype=dtype, device=device)
    s = None if fused_normalizer else torch.randn(size=(b, h, D), dtype=torch.float32, device=device).abs()
    if requires_grad:
        Q, K, V, log_G, initial_state, s = tree_map(
            lambda x: x.clone().detach().requires_grad_(True) if x is not None else None, (Q, K, V, log_G, initial_state, s))
    return dict(Q=Q, K=K, V=V, log_G=log_G, 
                initial_state=initial_state, sum_of_keys=s,
                deg=deg, scale=scale,
                switch_over_seq_len=switch_over_seq_len)

def input_properties(b=2, t=1024, h=8, d=32, qhead_ratio=1, dtype=torch.float16, device='cuda', gating=False,
                  chunk_size=None, deg=2, requires_grad=False, seed=42, scale=1.0, switch_over_seq_len=None):
    return dict(
        Q=((b, t, h * qhead_ratio, d), dtype, device),
        K=((b, t, h, d), dtype, device),
        V=((b, t, h, d), dtype, device)) | (dict(log_G=((b, t, h), torch.float32, device)) if gating else {})

def inference_input_properties(b=2, tq=1, tk=1024, h=8, d=32, qhead_ratio=1, dtype=torch.float16, device='cuda', gating=False,
                  chunk_size=None, deg=2, requires_grad=False, seed=42, scale=1.0, initial_state=False, switch_over_seq_len=512, fused_normalizer=True):
    D = sympow_dim(d, deg, d_tile=default_d_tile(d, deg)) if fused_normalizer else compute_expanded_dim(d, deg=deg)
    if tk > 0:
        return dict(
            Q=((b, tq, h * qhead_ratio, d), dtype, device),
            K=((b, tk, h, d), dtype, device),
            V=((b, tk, h, d), dtype, device)) | (dict(log_G=((b, tk, h), torch.float32, device)) if gating else {}) | (dict(initial_state=((b, h, D, d), dtype, device)) if initial_state else {}) | (dict(sum_of_keys=((b, h, D), torch.float32, device)) if not fused_normalizer else {})
    else:
        return dict(Q = ((b, tq, h * qhead_ratio, d), dtype, device),
                    initial_state=((b, h, D, d), dtype, device) if initial_state else None,
                    sum_of_keys=((b, h, D), torch.float32, device) if not fused_normalizer else None)

def output_properties(b=2, t=1024, h=8, d=32, qhead_ratio=1, dtype=torch.float16, device='cuda', gating=False,
                  chunk_size=None, deg=2, requires_grad=False, seed=42, scale=1.0, switch_over_seq_len=None):
    return dict(
        Y=((b, t, h * qhead_ratio, d), dtype, device)
    )

def inference_output_properties(b=2, tq=1, tk=1024, h=8, d=32, qhead_ratio=1, dtype=torch.bfloat16, device='cuda', gating=False,
                  chunk_size=None, deg=2, requires_grad=False, seed=42, scale=1.0, initial_state=False, switch_over_seq_len=512, fused_normalizer=True):
    D = sympow_dim(d, deg, d_tile=default_d_tile(d, deg)) if fused_normalizer else compute_expanded_dim(d, deg=deg)
    has_state = (initial_state or (chunk_size is not None and tk % chunk_size == 0 and tk >= switch_over_seq_len))
    return dict(
        Y=((b, tq, h * qhead_ratio, d), dtype, device),
        state=((b, h, D, d), dtype, device) if has_state else None,
    ) | (dict(sum_of_keys=((b, h, D), torch.float32, device) if has_state else None) if not fused_normalizer else {})