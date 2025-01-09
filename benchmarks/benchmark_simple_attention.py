# Install the newest triton version with
# pip install "git+https://github.com/openai/triton.git#egg=triton&subdirectory=python"
import pickle
import math
import torch
from datetime import date
torch._dynamo.config.cache_size_limit = 64
import torch.nn as nn
import torch.nn.functional as F

import os
from einops import rearrange, repeat
from torch.utils._pytree import tree_map


from benchmark import (
    benchmark_forward,
    benchmark_backward,
)
from power_attention.timing_utils import get_compiled_versions, estimate_runtime

from power_attention.power_full import PowerAttentionKernel
from power_attention.attention import symmetric_power_attention

from flash_attn import flash_attn_func

FORCE_RERUN_TODAY = True

def efficiency(tokens, time):
    return (tokens / time) if not math.isnan(time) else 0.0

def time_fn(func, *args, repeats=10, which='fb', **kwargs):
    f, b, fb = get_compiled_versions(func, *args, **kwargs)
    num1, num2 = repeats, repeats * 2
    if which == 'f':
        t = estimate_runtime(f, num1=num1, num2=num2)
    elif which == 'b':
        t = estimate_runtime(b, num1=num1, num2=num2)
    elif which == 'fb':
        t = estimate_runtime(fb, num1=num1, num2=num2)
    else:
        raise ValueError(f"Invalid value for which: {which}")
    return t

today = "2024-11-10"
warmup = 5
repeats = 30
device = 'cuda'
dtype = torch.float16
which = 'bwd'

bs_seqlen_vals = [(2**16 // seqlen, seqlen) for seqlen in [2**i for i in range(10, 17)]]
headdim_vals = [64] # [32, 64]
dim = 64
dropout_p = 0.0

methods = (["Flash2", f"Power Full Interface", "SDPA"])
speeds = {}

if True:
    times = {}
    for headdim in headdim_vals:
        for batch_size, seqlen in bs_seqlen_vals:
            config = (headdim, batch_size, seqlen)
            nheads = dim // headdim

            print(f"### headdim={headdim}, batch_size={batch_size}, seqlen={seqlen} ###")

            # Flash2
            method_key = "Flash2"
            if method_key in methods:
                if method_key in speeds and config in speeds[method_key]:
                    print(f"Skipping {method_key} - already in results")
                else:
                    qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, device=device, dtype=dtype,
                                        requires_grad=True)
                    q, k, v = qkv.unbind(dim=2)
                    times[config, method_key] = time_fn(
                        flash_attn_func, q, k, v, dropout_p, causal=True, repeats=repeats, which=which
                    )

            # SDPA
            method_key = "SDPA"
            if method_key in methods:
                if method_key in speeds and config in speeds[method_key]:
                    print(f"Skipping {method_key} - already in results")
                else:
                    qkv = qkv.detach().requires_grad_(True)
                    q, k, v = qkv.unbind(dim=2)
                    q = rearrange(q, 'b s h d -> b h s d')
                    k = rearrange(k, 'b s h d -> b h s d') 
                    v = rearrange(v, 'b s h d -> b h s d')
                    times[config, method_key] = time_fn(
                        F.scaled_dot_product_attention, q, k, v, is_causal=True,
                        repeats=repeats, which=which
                    )

            # Power Attention
            method_key = f"Power Full Interface"
            if method_key in methods:
                if (not FORCE_RERUN_TODAY) and method_key in speeds and config in speeds[method_key]:
                    print(f"Skipping {method_key} - already in results")
                else:
                    q, k, v = [torch.randn(batch_size, 1, seqlen, nheads, headdim, device=device, dtype=dtype,
                                        requires_grad=True) for _ in range(3)]
                    # q, k, v = tree_map(
                    #     lambda x: x[:, None, ...], (q, k, v),
                    # )
                    r = 5 + torch.randn(batch_size, 1, seqlen, nheads, device=device, dtype=torch.float32, requires_grad=True)
                    # power_attn = PowerAttentionKernel(headdim, 2, 1e-5, dtype)
                    power_attn = lambda Q, K, V, R: symmetric_power_attention(Q, K, V, R, 2, None, 1e-5)
                    times[config, method_key] = time_fn(
                        power_attn, q, k, v, r,
                        repeats=repeats
                    )

            for method_key in methods:
                if method_key not in speeds:
                    speeds[method_key] = {}
                if config not in speeds[method_key]:
                    speeds[method_key][config] = {}
                if (config, method_key) not in times:
                    continue
                speeds[method_key][config] = efficiency(
                    batch_size * seqlen,
                    times[config, method_key]
                )
                print(
                    f"{method_key} "
                    f"{which}: {speeds[method_key][config] / 1e6:.2f} Mtok/s"
                )

print('Plotting results...')
from bokeh.plotting import figure
from bokeh.palettes import Spectral4
from bokeh.io import save, output_file
import os

# Prepare data
x_vals = [sl for _, sl in bs_seqlen_vals]

# Create plots directory if it doesn't exist
plots_dir = os.path.expanduser('~/power_attention_benchmark_plots')
os.makedirs(plots_dir, exist_ok=True)

# Get distinct colors for all methods
colors = Spectral4[:len(methods)]

# Create a plot for each head dimension
for head_dim in headdim_vals:
    p = figure(
        title=f'Attention Speed Comparison (head_dim={head_dim}, {which})',
        x_axis_label='Sequence Length',
        y_axis_label='tokens / second',
        x_axis_type='log',
        x_range=(3500, max(x_vals)),
        y_range=(-100, 2_200_000),
        width=1000,
        height=500
    )
    # Plot all methods with distinct colors
    for method, color in zip(methods, colors):
        y_vals = []
        for bs, sl in bs_seqlen_vals:
            config = (head_dim, bs, sl)
            if config in speeds[method]:
                y_vals.append(speeds[method][config])
            else:
                y_vals.append(float('nan'))
        p.line(x_vals, y_vals, line_color=color, line_width=3,
               legend_label=method)

    p.line(x_vals, [1_200_000] * len(x_vals), line_color='black', line_dash='dashed',
           legend_label='Theoretically Achievable', line_width=4)
    p.legend.click_policy = 'hide'
    p.add_layout(p.legend[0], 'right')
    output_file(os.path.join(plots_dir, f'simple_attention_benchmark_head{head_dim}.html'),
                title=f'Attention Speed Comparison (head_dim={head_dim})')
    save(p)
    # Create ratio plot
    p_ratio = figure(
        title=f'Power Attention Advantage vs Flash Attention (head_dim={head_dim})',
        x_axis_label='Sequence Length',
        y_axis_label='Speedup multiplier',
        x_axis_type='log',
        width=600,
        height=400
    )
    # Calculate and plot ratio between today's power attention and flash
    ratio_vals = []
    today_power = f"Power Full Interface"
    for bs, sl in bs_seqlen_vals:
        config = (head_dim, bs, sl)
        if config in speeds[today_power] and config in speeds["Flash2"]:
            ratio = speeds[today_power][config] / speeds["Flash2"][config]
            ratio_vals.append(ratio)
        else:
            ratio_vals.append(float('nan'))
    p_ratio.line(x_vals, ratio_vals, line_color='navy', line_width=3)
    # Add a reference line at y=1
    p_ratio.line(x_vals, [1.0] * len(x_vals), line_color='red', 
                 line_dash='dashed', line_width=1)
    output_file(os.path.join(plots_dir, f'simple_attention_ratio_head{head_dim}.html'),
                title=f'Power vs Flash Attention Speed Ratio (head_dim={head_dim})')
    save(p_ratio)