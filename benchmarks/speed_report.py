import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F
torch._dynamo.config.cache_size_limit = 64
from einops import rearrange
from torch.utils._pytree import tree_map

from power_attention import attention as symmetric_power_attention
from power_attention import chunk_state as symmetric_power_chunk_state
from power_attention import query_state as symmetric_power_query_state
from power_attention import discumsum
from power_attention import compute_expanded_dim
from power_attention import power_full
from power_attention.timing_utils import get_compiled_versions, estimate_runtime


def create_test_tensors(batch_size=4, seqlen=4096, nheads=32, headdim=64, device='cuda', dtype=torch.float16):
    """Create test tensors for power attention benchmarking."""
    Q, K, V = (torch.randn(size=(batch_size, seqlen, nheads, headdim), dtype=dtype, device=device, requires_grad=True)/(headdim**.5) for _ in range(3))
    log_G = torch.zeros((batch_size, seqlen, nheads), dtype=torch.float32, device=device, requires_grad=True) - .01 if gating else None
    return Q, K, V, log_G

def time_fn(func, *args, repeats=10, **kwargs):
    f, b, fb = get_compiled_versions(func, *args, **kwargs)
    num1, num2 = repeats, repeats * 2
    time_f = estimate_runtime(f, num1=num1, num2=num2)
    time_b = estimate_runtime(b, num1=num1, num2=num2)
    time_fb = estimate_runtime(fb, num1=num1, num2=num2)
    return time_f, time_b, time_fb

def power_attention_speed_report(batch_size, seqlen, nheads, headdim, dtype, deg, chunk_size, gating, repeats=30, device='cuda'):
    # Create test tensors
    Q, K, V, log_G = create_test_tensors(batch_size, seqlen, nheads, headdim, device, dtype)
    if not gating:
        log_G = None

    # Create kernel
    times = {}

    # Full forward + backward
    times['full'] = time_fn(power_full, Q, K, V, log_G, chunk_size=chunk_size, repeats=repeats)

    # Unpack the internal logic of the kernel interface in order to benchmark chunk_state
    b, t, h, d = batch_size, seqlen, nheads, headdim
    c = chunk_size
    n = t // c
    D = compute_expanded_dim(d, 2)

    Q, K, V = (A.view(b, n, c, h, d) for A in (Q, K, V))
    log_G_intrachunk_accum = log_G.view(b, n, c, h).to(torch.float32).cumsum(2) if gating else None

    if gating:
        log_discount_weights = (log_G_intrachunk_accum.narrow(2, c-1, 1) - log_G_intrachunk_accum) / deg
        cs_K = K * torch.exp(log_discount_weights).unsqueeze(-1).to(K.dtype)
    else:
        cs_K = K
    cs_K, V = cs_K.contiguous(), V.contiguous()
    print("timing chunk_state")
    # Benchmark chunk_state
    times['chunk_state'] = time_fn(symmetric_power_chunk_state, cs_K, V, deg, repeats=repeats)

    # Unpack the internal logic of the kernel interface in order to benchmark accumulate_state
    S, s = symmetric_power_chunk_state(cs_K, V, deg)
    log_G_chunk_sum = log_G_intrachunk_accum[:,:,-1].contiguous() if gating else None
    print("timing accumulate_state")
    # Benchmark accumulate_state
    times['accumulate_state'] = tuple(a+b for a,b in zip(time_fn(discumsum, S, log_G_chunk_sum, repeats=repeats), time_fn(discumsum, s, log_G_chunk_sum, repeats=repeats)))

    # Benchmark local attention
    print("timing local_attention")
    Q, K, V = (A.view(b * n, c, h, d) for A in (Q, K, V))
    times['local_attention'] = time_fn(symmetric_power_attention, Q, K, V, None, deg, float(D), 1e-5, False)

    # Unpack the internal logic of the kernel interface in order to benchmark query_state
    S = discumsum(S, log_G_chunk_sum)
    s = discumsum(s, log_G_chunk_sum)
    # Query state
    Q = Q.contiguous()
    S = S.narrow(1, 0, n).contiguous()
    s = s.narrow(1, 0, n).contiguous()
    log_G_intrachunk_accum = log_G_intrachunk_accum.view(b * n, c, h)
    attn_Y, attn_y = symmetric_power_attention(Q, K, V, log_G_intrachunk_accum, deg, float(D), 1e-5, False)
    # Benchmark query_state
    print("timing query_state")
    Q, K, V = (A.view(b, n, c, h, d) for A in (Q, K, V))
    attn_Y = attn_Y.view(b, n, c, h, d)
    attn_y = attn_y.view(b, n, c, h)
    times['query_state'] =  time_fn(symmetric_power_query_state, Q, S, s, attn_Y, attn_y, deg, 1., 1e-5, False)
    return times

import numpy as np
import os
from bokeh.io import output_file, save
from bokeh.layouts import gridplot, row
from bokeh.plotting import figure
from bokeh.palettes import Spectral6
from bokeh.models import Legend

# Configuration
plots_dir = os.path.expanduser('~/power_attention_benchmark_plots')
device = torch.device('cuda')
dtype = torch.float16
dim = 2048
headdim = 64
nheads = dim // headdim
deg = 2
ε = 1e-5
repeats = 5
chunk_size = 2048
gating = True

# Test different sequence lengths
seqlen_vals = [2**i for i in range(13, 17)]
bs_seqlen_vals = [(2**16 // seqlen, seqlen) for seqlen in seqlen_vals]

# Collect results
all_times = {}
for batch_size, seqlen in bs_seqlen_vals:
    print(f"Running benchmark for batch_size={batch_size}, seqlen={seqlen}")
    times = power_attention_speed_report(batch_size, seqlen, nheads, headdim, dtype, deg, chunk_size, gating, repeats, device)
    all_times[seqlen] = times

# Prepare data for plotting
components = ['chunk_state', 'accumulate_state', 'local_attention', 'query_state']
x = seqlen_vals

# Calculate y range across all plots
def get_times(thing, which_timing):
    if which_timing == 'fwd':
        return thing[0]
    elif which_timing == 'bwd':
        return thing[1]
    elif which_timing == 'fwd+bwd':
        return thing[2]
    else:
        raise ValueError(f"Invalid timing: {which_timing}")

y_range = (0, max([get_times(all_times[seqlen]['full'], 'fwd+bwd') for seqlen in seqlen_vals]))

def plot_stacked(which_timing, title, chunk_size):
    p = figure(
        title=title,
        x_axis_label='Sequence Length',
        y_axis_label='Time (ms)', 
        x_axis_type='log',
        width=500,
        height=400,
        y_range=y_range,
        tooltips=[('Component', '@component'), ('Time', '@time{0.000} ms')]
    )
    
    y_components = np.array([[get_times(all_times[seqlen][comp], which_timing) for seqlen in seqlen_vals] for comp in components])
    y_total = np.array([get_times(all_times[seqlen]['full'], which_timing) for seqlen in seqlen_vals])
    
    # Create stacked areas
    bottom = np.zeros(len(x))
    renderers = []
    for i, (comp, color) in enumerate(zip(components, Spectral6)):
        top = bottom + y_components[i]
        source = {
            'x': x,
            'bottom': bottom,
            'top': top,
            'component': [comp] * len(x),
            'time': y_components[i]
        }
        r = p.varea('x', 'bottom', 'top', source=source, alpha=0.8, color=color)
        renderers.append((r, comp))
        bottom = top
        
    # Add dotted line for total
    source = {
        'x': x,
        'y': y_total,
        'component': ['Total'] * len(x),
        'time': y_total
    }
    r = p.line('x', 'y', source=source, line_dash='dotted', line_width=2, color='black')
    renderers.append((r, 'Total'))
        
    p.grid.grid_line_alpha = 0.2
    return p, renderers

# Create three subplots
p1, renderers1 = plot_stacked('fwd+bwd', 'Forward + Backward Time', chunk_size)
p2, _ = plot_stacked('fwd', 'Forward Time', chunk_size)
p3, _ = plot_stacked('bwd', 'Backward Time', chunk_size)

# Create legend items
legend_items = [(comp, [r]) for r, comp in renderers1]
legend = Legend(items=legend_items)
legend.label_text_font_size = '8pt'

# Add legend to first plot
p1.add_layout(legend, 'left')

# Arrange plots in a grid
grid = gridplot([[p1, p2, p3]], sizing_mode="stretch_width")

output_file(os.path.join(plots_dir, f'power_attention_speed_report_head{headdim}.html'),
            title=f'Power Attention Speed Report (head_dim={headdim})')
save(grid)