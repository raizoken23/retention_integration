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

from state_kernel.timing_utils import get_compiled_versions, estimate_runtime

from packages.state_kernel.state_kernel.power_full import power_full

from flash_attn import flash_attn_func

FORCE_RERUN_TODAY = True

def efficiency(tokens, time):
    return (tokens / time * 1000) if not math.isnan(time) else 0.0

def attention_pytorch(qkv, dropout_p=0.0, causal=True):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
    """
    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    q = rearrange(q, 'b t h d -> (b h) t d')
    k = rearrange(k, 'b s h d -> (b h) d s')
    softmax_scale = 1.0 / math.sqrt(d)
    # Preallocate attn_weights for `baddbmm`
    scores = torch.empty(batch_size * nheads, seqlen, seqlen, dtype=qkv.dtype, device=qkv.device)
    scores = rearrange(torch.baddbmm(scores, q, k, beta=0, alpha=softmax_scale),
                       '(b h) t s -> b h t s', h=nheads)
    if causal:
        # "triu_tril_cuda_template" not implemented for 'BFloat16'
        # So we have to construct the mask in float
        causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
        # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
        scores = scores + causal_mask.to(dtype=scores.dtype)
    attention = torch.softmax(scores, dim=-1)
    attention_drop = F.dropout(attention, dropout_p)
    output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    return output.to(dtype=qkv.dtype)


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

today = "2024-11-14"
warmup = 5
repeats = 10
device = 'cuda'
dtype = torch.float16
which = 'fb' # forward + backward

bs_seqlen_vals = [(2**16 // seqlen, seqlen) for seqlen in [2**i for i in range(11, 17)]]
headdim_vals = [64] # [32, 64]
dim = 2048
dropout_p = 0.0
chunk_size = 1024
initial_attention_chunk_n = 4
gating = False

print(f'{bs_seqlen_vals=} {headdim_vals=} {dim=} {dropout_p=} {chunk_size=} {initial_attention_chunk_n=} {gating=}')

methods = ["Flash2", "SDPA", f"Power (p=2), {today}"]
# methods = (["Pytorch", "Flash2", f"Power (p=2), {today}", "SDPA"])

# Load existing results if they exist
speeds = {}
results_path = os.path.expanduser('~/power_attention_benchmark.pkl')
if os.path.exists(results_path):
    with open(results_path, 'rb') as fp:
        speeds = pickle.load(fp)

if True:
    times = {}
    for headdim in headdim_vals:
        for batch_size, seqlen in bs_seqlen_vals:
            config = (headdim, batch_size, seqlen)
            nheads = dim // headdim

            print(f"### headdim={headdim}, batch_size={batch_size}, seqlen={seqlen}, gating={gating}, chunk_size={chunk_size}, initial_attention_chunk_n={initial_attention_chunk_n} ###")

            # Flash2
            method_key = "Flash2"
            if method_key in methods:
                if method_key in speeds and config in speeds[method_key]:
                    print(f"Skipping {method_key} - already in results")
                else:
                    Q, K, V = (torch.randn(size=(batch_size, seqlen, nheads, headdim), dtype=dtype, device='cuda', requires_grad=True)/(headdim**.5) for _ in range(3))
                    times[config, method_key] = time_fn(
                        flash_attn_func, Q, K, V, dropout_p, causal=True, repeats=repeats, which=which
                    )


            # SDPA
            method_key = "SDPA"
            if method_key in methods:
                if method_key in speeds and config in speeds[method_key]:
                    print(f"Skipping {method_key} - already in results")
                else:
                    Q, K, V = (torch.randn(size=(batch_size, seqlen, nheads, headdim), dtype=dtype, device='cuda', requires_grad=True)/(headdim**.5) for _ in range(3))
                    Q, K, V = (rearrange(A, 'b s h d -> b h s d') for A in (Q, K, V))
                    times[config, method_key] = time_fn(
                        F.scaled_dot_product_attention, Q, K, V, is_causal=True,
                        repeats=repeats, which=which
                    )

            # Power Attention
            method_key = f"Power (p=2), {today}"
            if method_key in methods:
                if (not FORCE_RERUN_TODAY) and method_key in speeds and config in speeds[method_key]:
                    print(f"Skipping {method_key} - already in results")
                else:
                    Q, K, V = (torch.randn(size=(batch_size, seqlen, nheads, headdim), dtype=dtype, device='cuda', requires_grad=True)/(headdim**.5) for _ in range(3))
                    log_G = torch.zeros((batch_size, seqlen, nheads), dtype=torch.float32, device='cuda', requires_grad=True) - .01 if gating else None
                    times[config, method_key] = time_fn(power_full,
                        Q, K, V, log_G, chunk_size=chunk_size, initial_attention_chunk_n=initial_attention_chunk_n,
                        repeats=repeats, which=which
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

    print('Pickling results...')
    with open(results_path, 'wb') as fp:
        pickle.dump(speeds, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print('Done.')

print('Plotting results...')
from bokeh.plotting import figure
from bokeh.palettes import Spectral4, Viridis
from bokeh.io import save, output_file
import os

# Prepare data
x_vals = [sl for _, sl in bs_seqlen_vals]

# Split methods into power and non-power
power_dates = sorted([k for k in speeds.keys() if k.startswith('Power')])
other_methods = [m for m in methods if not m.startswith('Power')]

# Create plots directory if it doesn't exist
plots_dir = os.path.expanduser('~/power_attention_benchmark_plots')
os.makedirs(plots_dir, exist_ok=True)

# Get distinct colors for non-power methods
other_colors = Spectral4[:len(other_methods)]

# Create a plot for each head dimension
for head_dim in headdim_vals:
    p = figure(
        title=f'Attention Speed Comparison (head_dim={head_dim}, {which}, {gating=})',
        x_axis_label='Sequence Length',
        y_axis_label='tokens / second',
        x_axis_type='log',
        x_range=(3500, max(x_vals)),
        y_range=(-100, 2_200_000),
        width=1000,
        height=500
    )
    # Plot non-power methods with distinct colors and dotted lines
    for method, color in zip(other_methods, other_colors):
        if method == "Pytorch" or method == "SDPA":
            continue
        y_vals = []
        for bs, sl in bs_seqlen_vals:
            config = (head_dim, bs, sl)
            if config in speeds[method]:
                y_vals.append(speeds[method][config])
            else:
                y_vals.append(float('nan'))
        p.line(x_vals, y_vals, line_color=color, line_dash='dotted', 
               legend_label=method, line_width=3)
    # Plot power methods with color gradient
    power_colors = Viridis[len(power_dates)]
    for i, (date, color) in enumerate(zip(power_dates, power_colors)):
        y_vals = []
        for bs, sl in bs_seqlen_vals:
            config = (head_dim, bs, sl)
            if config in speeds[date]:
                y_vals.append(speeds[date][config])
            else:
                y_vals.append(float('nan'))
        p.line(x_vals, y_vals, line_color=color,
               legend_label=f"Power Attention (p=2), Day {i+4}", line_width=4)
    p.line(x_vals, [1_200_000] * len(x_vals), line_color='black', line_dash='dashed',
           legend_label='Theoretically Achievable', line_width=4)
    p.legend.click_policy = 'hide'
    p.add_layout(p.legend[0], 'right')
    output_file(os.path.join(plots_dir, f'power_attention_benchmark_head{head_dim}.html'),
                title=f'Attention Speed Comparison (head_dim={head_dim})')
    save(p)
    # Create ratio plot
    p_ratio = figure(
        title=f'Power Attention Advantage vs Flash Attention (head_dim={head_dim}, {which})',
        x_axis_label='Sequence Length',
        y_axis_label='Speedup multiplier',
        x_axis_type='log',
        y_axis_type='log',
        width=600,
        height=400
    )
    # Calculate and plot ratio between today's power attention and flash
    ratio_vals = []
    theoretical_ratio_vals = []
    today_power = f"Power (p=2), {today}"
    for bs, sl in bs_seqlen_vals:
        config = (head_dim, bs, sl)
        if config in speeds[today_power] and config in speeds["Flash2"]:
            ratio = speeds[today_power][config] / speeds["Flash2"][config]
            ratio_vals.append(ratio)
            # Calculate theoretical maximum ratio
            theoretical_ratio = 1900000 / speeds["Flash2"][config]
            theoretical_ratio_vals.append(theoretical_ratio)
        else:
            ratio_vals.append(float('nan'))
            theoretical_ratio_vals.append(float('nan'))
    p_ratio.line(x_vals, ratio_vals, line_color='navy', line_width=3)
    # Add a reference line at y=1
    p_ratio.line(x_vals, [1.0] * len(x_vals), line_color='red', 
                 line_dash='dashed', line_width=1)
    # Add theoretical maximum speedup line
    # p_ratio.line(x_vals, theoretical_ratio_vals, line_color='black',
    #              line_dash='dotted', line_width=1)
    output_file(os.path.join(plots_dir, f'power_attention_ratio_head{head_dim}.html'),
                title=f'Power vs Flash Attention Speed Ratio (head_dim={head_dim})')
    save(p_ratio)
