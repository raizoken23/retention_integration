import torch
from retention.vidrial_fused import power_retention
from retention.create_inputs import create_inputs
from fla.layers.gla import GatedLinearAttention
from fla.layers.rwkv7 import RWKV7Attention
from perf._timing import estimate_runtime, get_compiled_version, sanitize_kwargs
from flash_attn.flash_attn_interface import flash_attn_func
from vidrial.py_utils.common import default_d_tile
from vidrial.kernels.sympow_mma.dimensions import sympow_dim
from vidrial.jit.settings import settings, PickBest
import pandas as pd
import logging
from perf.ablations.model import PowerAttention




def flops_estimate(b, t, h, d, qhead_ratio, deg, chunk_size):
    D = sympow_dim(d, deg, d_tile=default_d_tile(d, deg))
    attention_flops = (qhead_ratio * t * d * 2 + qhead_ratio * d * t * 2) * b * h
    query_state_flops = (qhead_ratio * D * d * 2) * b * h
    update_state_flops = 0 if t < chunk_size else (D * d * 2 *t)
    return attention_flops + query_state_flops + update_state_flops


def measure_power_time(b, t, h, qhead_ratio, d, chunk_size, deg, gating, dtype, **kwargs):
    model = PowerAttention(
        num_heads=h,
        hidden_size=h * d,
        chunk_size=chunk_size,
        degree=deg,
        head_size=d,
        qhead_ratio=qhead_ratio,
        gating=gating,
        device='cuda',
        dtype=dtype,
        kernel='power',
    ).to('cuda').to(dtype)
    model.train()
    hidden_states = torch.randn(b, t, h * d, device="cuda", dtype=dtype, requires_grad=True)
    time = estimate_runtime(get_compiled_version(lambda hidden_states: model(hidden_states=hidden_states), {'hidden_states': hidden_states}, direction='fwd+bwd', compile=False), num1=2, num2=10)
    return time

def measure_flash_time(b, t, h, qhead_ratio, d, chunk_size, deg, gating, dtype, **kwargs):
    model = PowerAttention(
        num_heads=h,
        hidden_size=h * d,
        chunk_size=chunk_size,
        degree=deg,
        head_size=d,
        qhead_ratio=qhead_ratio,
        gating=False,
        device='cuda',
        dtype=dtype,
        kernel='flash',
    ).to('cuda').to(dtype)
    model.train()
    hidden_states = torch.randn(b, t, h * d, device="cuda", dtype=dtype, requires_grad=True)
    time = estimate_runtime(get_compiled_version(lambda hidden_states: model(hidden_states=hidden_states), {'hidden_states': hidden_states}, direction='fwd+bwd', compile=False), num1=2, num2=10)
    return time

def measure_rwkv7_time(b, t, h, qhead_ratio, d, chunk_size, deg, gating, dtype, **kwargs):
    h = h * qhead_ratio # rwkv7 doesn't support GQA
    hidden_size = h * d
    head_dim = d
    model = RWKV7Attention(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=h,
        layer_idx=0,
        num_hidden_layers=2, 
        **kwargs
    ).to('cuda').to(dtype)
    model.train()
    hidden_states = torch.randn(b, t, hidden_size, device="cuda", dtype=dtype, requires_grad=True)
    time = estimate_runtime(get_compiled_version(lambda hidden_states: model(hidden_states=hidden_states)[0], {'hidden_states': hidden_states}, direction='fwd+bwd', compile=False), num1=2, num2=10)
    return time

def measure_gla_time(b, t, h, qhead_ratio, d, chunk_size, deg, gating, dtype, **kwargs):
    model = GatedLinearAttention(
        mode='chunk',
        hidden_size=h * d * qhead_ratio,
        expand_k=1.0,
        expand_v=1.0,
        num_heads=h*qhead_ratio,
        num_kv_heads=h,
        layer_idx=0,
        **kwargs
    ).to('cuda').to(dtype)
    model.train()
    hidden_states = torch.randn(b, t, h * d * qhead_ratio, device="cuda", dtype=dtype, requires_grad=True)
    time = estimate_runtime(get_compiled_version(lambda hidden_states: model(hidden_states=hidden_states)[0], {'hidden_states': hidden_states, 'use_cache': False}, direction='fwd+bwd', compile=False), num1=2, num2=10)
    return time

def compare():
    # install from https://github.com/flashinfer-ai/flashinfer
    df = []
    b, h, d, chunk_size, deg, gating, dtype, switch_over_seq_len = 1, 8, 64, 64, 2, True, torch.bfloat16, 512
    print(f"Measuring runtime for {b=} {h=} {d=} {chunk_size=} {deg=} {gating=} {dtype=}")
    logging.basicConfig(level=logging.ERROR)
    qhead_ratio = 8

    with settings.set(policy=PickBest):
        for t in [128, 512, 2048, 8192, 32768, 65536]:
            print(f"========== {t=} ==========")
            gla_measurement = measure_gla_time(b=b, t=t, h=h, qhead_ratio=qhead_ratio, d=d, chunk_size=chunk_size, deg=deg, gating=gating, dtype=dtype)
            power_measurement = measure_power_time(b=b, t=t, h=h, d=d, qhead_ratio=qhead_ratio, deg=deg, chunk_size=chunk_size, dtype=dtype, gating=gating)
            rwkv7_measurement = measure_rwkv7_time(b=b, t=t, h=h, qhead_ratio=qhead_ratio, d=d, chunk_size=chunk_size, deg=deg, gating=gating, dtype=dtype) if t < 65536 else float('inf') # RWKV7 doesn't work 65536 for some reason, didn't bother investigating
            flash_measurement = measure_flash_time(b=b, t=t, h=h, qhead_ratio=qhead_ratio, d=d, chunk_size=chunk_size, deg=deg, gating=gating, dtype=dtype)
            df.append({
                'FlashAttention (ms)': flash_measurement,
                'RWKV7 (ms)': rwkv7_measurement,
                'GLA (ms)': gla_measurement,
                'Power (ms)': power_measurement,
                'RWKV7 Speedup': flash_measurement / rwkv7_measurement,
                'GLA Speedup': flash_measurement / gla_measurement,
                'Power Speedup': flash_measurement / power_measurement,
                'Context': t,
            })

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    df = pd.DataFrame(df)
    print("\nFull results table:")
    print(df.to_string())
    
    # Create pivoted table
    pivoted_df = df.set_index('Context')[['FlashAttention (ms)', 'RWKV7 Speedup', 'GLA Speedup', 'Power Speedup']].T
    print("\nPivoted table (speedup over flashinfer):")
    print(pivoted_df.to_string())
    
    # Print raw data
    print("\nRaw data:")
    print(df.to_dict('records'))

  
if __name__ == '__main__':
    compare()
