import math
import os
import sys
import time
import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark
from einops import rearrange
from flash_attn_manifest.flash_attn_interface import flash_attn_func
from power_attention import __version__ as power_attention_version
from power_attention.power_full import PowerAttentionKernel
from power_attention.attention import symmetric_power_attention
from torch.autograd.profiler import record_function
from torch.autograd.profiler_util import EventList
from torch.profiler import ProfilerActivity, profile
from torch.utils._pytree import tree_map

from power_attention.chunk_state import ExpandedDim

MEM_LIMIT = 80 * 1024 * 1024 * 1024  # 80 GB

class Config:
    def __init__(self, shape, chunk_size, p, stabilizer, dtype, acc_dtype, ε, seed=42, ref_dtype=None, critical_length=None, gating=True, expand=True):
        b, t, h, d = shape
        self.head_size = d
        self.ctx = t
        self.shape = shape  # [b t h d]
        self.chunk_size = chunk_size
        self.d = d
        self.p = p
        self.stabilizer = stabilizer
        self.dtype = dtype
        self.acc_dtype = acc_dtype
        self.ε = ε
        self.seed = seed
        self.ref_dtype = ref_dtype if ref_dtype is not None else dtype
        self.critical_length = critical_length
        self.gating = gating
        self.expand = expand

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


def benchmark_backward( # noqa: C901
    fn,
    *inputs,
    grad=None,
    repeats=10,
    desc='',
    verbose=False,
    amp=False,
    amp_dtype=torch.float16,
    **kwinputs,
):
    """Use Pytorch Benchmark on the backward pass of an arbitrary function."""
    if verbose:
        print(desc, '- Backward pass')

    with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp):
        y = fn(*inputs, **kwinputs)
        if type(y) is tuple:
            y = y[0]
    if grad is None:
        grad = torch.randn_like(y)
    else:
        if grad.shape != y.shape:
            msg = 'Grad shape does not match output shape'
            raise RuntimeError(msg)

    def f(*inputs, y, grad):
        # Set .grad to None to avoid extra operation of gradient accumulation
        for x in inputs:
            if isinstance(x, torch.Tensor):
                x.grad = None
        y.backward(grad, retain_graph=True)

    t = benchmark.Timer(
        stmt='f(*inputs, y=y, grad=grad)',
        globals={'f': f, 'inputs': inputs, 'y': y, 'grad': grad},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    return t, m

def mem_fit(ctx, chunk_size, D, head_size, batch_size, num_heads):
    n_chunks = ctx // chunk_size
    S_size = D * head_size * batch_size * num_heads * n_chunks * 2
    if S_size >= MEM_LIMIT:
        return False
    return True


def profile_whole(res, cfg: Config, args): # noqa: C901
    bwd = args.bwd
    verbose = args.verbose
    compile = args.compile

    if compile:
        symmetric_power_attention_ = torch.compile(symmetric_power_attention)
        kernel_ = torch.compile(kernel)
    else:
        symmetric_power_attention_ = symmetric_power_attention
        kernel_ = kernel

    torch.manual_seed(cfg.seed)
    kernel = PowerAttentionKernel(d=cfg.d, deg=cfg.p, ε=cfg.ε, dtype=cfg.dtype)
    benchmark_fn = benchmark_forward if not bwd else benchmark_backward
    # benchmark_fn = dumb_benchmark if not bwd else dumb_benchmark_backward

    Q, K, V = (torch.randn(cfg.shape, dtype=cfg.dtype, device='cuda') / (cfg.d**0.5) for _ in range(3))
    R = torch.randn(cfg.shape[:-1], dtype=torch.float32, device='cuda') * 5.0

    if bwd:
        Q.requires_grad_(True)
        K.requires_grad_(True)
        V.requires_grad_(True)
        R.requires_grad_(True)

    # the attention kernel needs [b n c h d]
    Q_attn, K_attn, V_attn = tree_map(
        lambda x: rearrange(x, 'b t h d -> b 1 t h d'),
        (Q, K, V),
    )

    def attn_fn():
        log_G = rearrange(F.logsigmoid(R).cumsum(1), 'b t h -> b 1 t h') if cfg.gating else None
        Y_attn, y_attn = symmetric_power_attention_(Q_attn, K_attn, V_attn, log_G, cfg.p, cfg.stabilizer, cfg.ε)
        return rearrange(Y_attn / y_attn[:, :, :, :, None], 'b n c h d -> b (n c) h d')

    def flash_attn_fn():
        return flash_attn_func(Q, K, V)

    def kernel_fn():
        return kernel_(Q, K, V, R, chunk_size=cfg.chunk_size, critical_length=cfg.critical_length)

    q, k, v = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

    def scaled_dot_product_attention_fn():
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)

    def wrap(fn):
        def wrapper():
            if bwd:
                return fn().mean()
            return fn()

        return wrapper

    _, m_chunk = benchmark_fn(wrap(kernel_fn), repeats=1, desc='Chunking')
    _, m_attn = benchmark_fn(wrap(attn_fn), repeats=1, desc='Sympow Attention')
    _, m_flash_attn = benchmark_fn(wrap(flash_attn_fn), repeats=1, desc='Flash Attention')
    _, m_sdpa = benchmark_fn(wrap(scaled_dot_product_attention_fn), repeats=1, desc='Scaled Dot Product Attention')

    if verbose:
        print(f'Sympow Attention: {m_attn.mean:4f} seconds')
        print(f'Chunking: {m_chunk.mean:4f} seconds')
        print(f'Flash Attention: {m_flash_attn.mean:4f} seconds')
        print(f'Scaled Dot Product Attention: {m_sdpa.mean:4f} seconds')

    res['ctx'].append(cfg.ctx)
    res['chunk_size'].append(cfg.chunk_size)
    res['p'].append(cfg.p)
    res['head_size'].append(cfg.head_size)

    res['Power Attention'].append(m_attn.mean)
    res['Power Recurrent'].append(m_chunk.mean)
    res['Flash Attention'].append(m_flash_attn.mean)
    res['SDPA'].append(m_sdpa.mean)

    return res


def just_run(cfg: Config, flavor, args): # noqa: C901
    torch.manual_seed(cfg.seed)
    os.environ['POWER_EXPAND'] = "1" if cfg.expand else "0"
    kernel = PowerAttentionKernel(d=cfg.d, deg=cfg.p, ε=cfg.ε, dtype=cfg.dtype)
    Q, K, V = (torch.randn(cfg.shape, dtype=cfg.dtype, device='cuda') / (cfg.d**0.5) for _ in range(3))
    R = torch.randn(cfg.shape[:-1], dtype=torch.float32, device='cuda') * 5.0

    if args.direction == 'bwd':
        Q.requires_grad_(True)
        K.requires_grad_(True)
        V.requires_grad_(True)
        R.requires_grad_(True)

    if flavor == 'power':
        fn = lambda: kernel(Q, K, V, R if cfg.gating else None, chunk_size=cfg.chunk_size, critical_length=cfg.critical_length)
    elif flavor == 'flash':
        def fn():
            return flash_attn_func(Q, K, V)
    elif flavor == 'sdpa':
        Q, K, V = tree_map(lambda x: x.transpose(1, 2), (Q, K, V))
        def fn():
            return torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=True)
    elif flavor == 'power_attn':
        Q_attn, K_attn, V_attn = tree_map(
            lambda x: rearrange(x, 'b t h d -> b 1 t h d'),
            (Q, K, V),
        )

        def _fn():
            if cfg.gating:
                log_G = rearrange(F.logsigmoid(R).cumsum(1), 'b t h -> b 1 t h')
                Y_attn, y_attn = symmetric_power_attention(Q_attn, K_attn, V_attn, log_G, cfg.p, cfg.stabilizer, cfg.ε)
            else:
                Y_attn, y_attn = symmetric_power_attention(Q_attn, K_attn, V_attn, None, cfg.p, cfg.stabilizer, cfg.ε)
            return rearrange(Y_attn / y_attn[:, :, :, :, None], 'b n c h d -> b (n c) h d')

        fn = _fn
    elif flavor == 'query_state_compare':
        def fn():
            from power_attention.attention import symmetric_power_attention
            from power_attention.query_state import symmetric_power_query_state
            from power_attention.chunk_state import ExpandedDim
            Q_attn, K_attn, V_attn = tree_map(
                lambda x: rearrange(x, 'b t h d -> b 1 t h d'),
                (Q, K, V),
            )
            Y_attn, y_attn = symmetric_power_attention(Q_attn, K_attn, V_attn, None, cfg.p, cfg.stabilizer, cfg.ε)
            b, t, h, d = cfg.shape
            D = ExpandedDim(d, cfg.p)
            n = t // cfg.chunk_size
            S = torch.randn(b, n, h, D, d, dtype=cfg.dtype, device=Q.device)
            s = torch.randn(b, n, h, D, dtype=torch.float32, device=Q.device)
            Q_chunk = rearrange(Q, 'b (n c) h d -> b n c h d', c=cfg.chunk_size)
            Y, y = symmetric_power_query_state(Q_chunk, S, s, cfg.p, cfg.stabilizer)
            return (Y + Y_attn) / (y + y_attn)[..., None]
    elif flavor == 'chunk_state_compare':
        def fn():
            from power_attention.chunk_state import symmetric_power_chunk_state
            from power_attention.attention import symmetric_power_attention
            Q_attn, K_attn, V_attn = tree_map(
                lambda x: rearrange(x, 'b t h d -> b 1 t h d'),
                (Q, K, V),
            )
            Y_attn, y_attn = symmetric_power_attention(Q_attn, K_attn, V_attn, None, cfg.p, cfg.stabilizer, cfg.ε)
            b, t, h, d = cfg.shape
            K_chunk, V_chunk = tree_map(lambda x: rearrange(x, 'b (n c) h d -> b n c h d', c=cfg.chunk_size), (K, V))
            S, s = symmetric_power_chunk_state(K_chunk, V_chunk, cfg.p)
            return (Y_attn / y_attn[:, :, :, :, None]).norm() + (S / s[..., None]).norm()


    if args.compile:
        fn = torch.compile(fn)
    os.environ['POWER_PROFILE'] = '1'
    with record_function('call::' + flavor):
        if args.direction == 'bwd':
            O = fn() # noqa: E741
            O.mean().backward()
        else:
            fn()
        torch.cuda.synchronize()
    del os.environ['POWER_PROFILE']


def profile_components(res, cfg, flavor, args):
    direction = args.direction
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
    ) as prof:
        for _ in range(8):
            just_run(cfg, flavor=flavor, args=args)
            prof.step()
    if flavor == 'power':
        prof.export_chrome_trace('trace.json')
    power_events = EventList(
        [
            x
            for x in prof.key_averages()
            if ((x.key.startswith('power::') or x.key.startswith('call::')) and x.device_type.name == 'CPU')
            or 'state_kernel::' in x.key
        ],
    )
    power_events_dict = {
        e.key: (e.cpu_time_total / 1e6) for e in power_events
    }
    if args.verbose:
        print(power_events.table(sort_by='cpu_time_total'))
    if flavor == 'power_attn':
        res['Power Attention'].append(power_events_dict['call::power_attn'])
        res['Power Attention CUDA'].append(power_events_dict['power::attention_forward_cuda'])
    elif flavor == 'power':
        res[f'Power Recurrent{"" if cfg.expand else "-unexpanded"}'].append(power_events_dict['call::power'])
        print(f'{cfg.ctx=}')
        if direction == 'bwd':
            print(f'accumulate bwd / fwd ratio:{power_events_dict["power::accumulate_state_backward"] / power_events_dict["power::accumulate_state"]}')
        res[f'Accumulation{"" if cfg.expand else "-unexpanded"}'].append(
            power_events_dict['power::accumulate_state'] + (0 if direction == 'fwd' else power_events_dict['power::accumulate_state_backward']),
        )
        res[f'Chunk States{"" if cfg.expand else "-unexpanded"}'].append(
            power_events_dict['power::chunk_state_cuda'] + (0 if direction == 'fwd' else power_events_dict['power::chunk_state_backward_cuda']),
        )
        res[f'Query States{"" if cfg.expand else "-unexpanded"}'].append(
            power_events_dict['power::query_states_fwd'] + (0 if direction == 'fwd' else power_events_dict['power::query_state_backward_cuda']),
        )
        res[f'Chunk Attention{"" if cfg.expand else "-unexpanded"}'].append(
            power_events_dict['power::attention_forward_cuda'] + (0 if direction == 'fwd' else power_events_dict['power::attention_backward_cuda']),
        )
    elif flavor == 'flash':
        res['Flash Attention'].append(power_events_dict['call::flash'])
    elif flavor == 'sdpa':
        res['SDPA'].append(power_events_dict['call::sdpa'])
    elif flavor == 'query_state_compare':
        res[f'Query States{"" if cfg.expand else "-unexpanded"}'].append(
            power_events_dict['power::query_states_fwd'] if direction=='fwd' else power_events_dict['power::query_state_backward_cuda'],
        )
        res[f'Power Attention{"" if cfg.expand else "-unexpanded"}'].append(
            power_events_dict['power::attention_forward_cuda'] if direction=='fwd' else power_events_dict['power::attention_backward_cuda'],
        )
    elif flavor == 'chunk_state_compare':
        res[f'Chunk States{"" if cfg.expand else "-unexpanded"}'].append(
            power_events_dict['power::chunk_state_cuda'] if direction=='fwd' else power_events_dict['power::chunk_state_backward_cuda'],
        )
        res[f'Power Attention{"" if cfg.expand else "-unexpanded"}'].append(
            power_events_dict['power::attention_forward_cuda'] if direction=='fwd' else power_events_dict['power::attention_backward_cuda'],
        )
    return res


def plot_absolute_perfs(data, direction, chunk_size, p, head_size, batch_size, num_heads, gating, critical_length, current_date):
    plt.figure(figsize=(10, 8))
    plt.plot(data['ctx'], data['Power Attention'], label='Power Attention', color='red')
    plt.plot(data['ctx'], data['Flash Attention'], label='Flash Attention', color='green')
    plt.plot(data['ctx'], data['Power Recurrent'], label='Power Recurrent', color='blue')
    plt.plot(data['ctx'], data['SDPA'], label='SDPA', color='purple')
    plt.title(
        f'Attention-{direction} vs. Power Recurrent-{direction} vs. SDPA-{direction} @ {current_date} \n'
        f'(Chunk: {chunk_size}, p: {p}, head_size: {head_size}, batch_size: {batch_size}, num_heads: {num_heads}, gating: {gating}, critical_length: {critical_length})',
    )
    plt.xlabel('Context Size')
    plt.xscale('log')
    plt.ylabel('Performance (s)')
    plt.yscale('log')
    plt.legend(loc='upper left')
    image_name = (f'benchmarks/absolute_perf_benchmark_'
                  f'{direction}_chunk_{chunk_size}_p_{p}_head_size_{head_size}_batch_size_{batch_size}_num_heads_{num_heads}_gating_{gating}_critical_length_{critical_length}_{current_date}.png')
    plt.savefig(image_name)
    return image_name


def plot_relative_perfs(data, args):
    head_size = args.head_size
    p = args.p
    chunk_size = args.chunk_size
    direction = args.direction
    batch_size = args.batch_size
    num_heads = args.num_heads
    gating = args.gating
    critical_length = args.critical_length

    D = ExpandedDim(head_size, p)
    fig, axes = plt.subplots(2, 1, figsize=(14,20))

    data.to_csv('benchmarks/relative_perf_benchmark.csv')

    # plt.plot(
    #     data['ctx'],
    #     (data['Flash Attention'] / data['Power Attention']).replace(0, np.nan),
    #     label='Power Attention speed-up over Flash Attention',
    #     color='red',
    #     marker='o',
    # )
    # plt.plot(data['ctx'], data['SDPA'] / data['Power Attention'], label='Power Attention speed-up over SDPA', color='orange', marker='o')
    theoretical_speedup = data['ctx'] / (2 * D + chunk_size)
    actual_speedup = data['Power Attention CUDA'] / (data['Query States'] + data['Chunk States'] + data['Chunk Attention'])
    speedup_gap = 1 - actual_speedup / theoretical_speedup
    axes[0].plot(
        data['ctx'],
        actual_speedup.replace(0, np.nan),
        label='Attention / (Query States + Chunk States + Chunk Attention) - Expanded',
        color='blue',
        marker='o',
    )
    if args.compare_expand:
        axes[0].plot(
            data['ctx'],
            (data['Power Attention CUDA'] / (data['Query States-unexpanded'] + data['Chunk States-unexpanded'] + data['Chunk Attention-unexpanded'])).replace(0, np.nan),
            label='Attention / (Query States + Chunk States + Chunk Attention) - Unexpanded',
            color='green',
            marker='o',
        )
    # axes[0].plot(
    #     data['ctx'],
    #     (data['SDPA'] / (data['Query States'] + data['Chunk States'])).replace(0, np.nan),
    #     label='Theoretical Power Recurrent speed-up over SDPA',
    #     color='purple',
    #     marker='o',
    # )
    # axes[0].plot(
    #     data['ctx'],
    #     ((data['Query States'] + data['Chunk States'] + data['Chunk Attention'] + data['Accumulation']) / data['Power Recurrent']).replace(0, np.nan),
    #     label='Kernel interface efficiency',
    #     color='pink',
    #     marker='o',
    # )
    axes[0].plot(data['ctx'], theoretical_speedup, label='Predicted Theoretical Speedup (all recurrent)', color='green', marker='o', linestyle='--')

    axes[0].set_yscale('log')
    axes[0].set_xscale('log')
    axes[0].axvline(x=D, color='black', linestyle='--')
    axes[0].text(D, 0, f'D = {D}', color='black', ha='right', va='bottom')
    axes[0].axvline(x=2*D, color='black', linestyle='--')
    axes[0].text(2*D, 0, f'2D = {2*D}', color='black', ha='right', va='bottom')
    axes[0].axhline(y=1, color='red', linestyle='--')
    axes[0].text(axes[0].get_xlim()[1], 1, 'y = 1', color='red', va='bottom', ha='left')

    axes[0].set_title(
        f'Relative Performance-{direction} \n(Chunk: {chunk_size}, p: {p}, head_size: {head_size}, batch_size: {batch_size}, '
        f'num_heads: {num_heads}, gating: {gating}, critical_length: {critical_length})',
    )

    axes[0].set_xscale('log')
    axes[0].set_xlabel('Context Size')
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Performance ratio')
    axes[0].legend(loc='upper left')
    image_name = (f'benchmarks/relative_perf_benchmark_{direction}_chunk_{chunk_size}_p_{p}'
                  f'_head_size_{head_size}_batch_size_{batch_size}_num_heads_{num_heads}_gating_{gating}_critical_length_{critical_length}_kernel_{power_attention_version}.png')
    
    axes[1].plot(data['ctx'], speedup_gap, label='Speedup Gap', color='blue', marker='o')
    if args.compare_expand:
        actual_speedup_unexpanded = data['Power Attention CUDA'] / (data['Query States-unexpanded'] + data['Chunk States-unexpanded'] + data['Chunk Attention-unexpanded'])
        speedup_gap_unexpanded = 1 - actual_speedup_unexpanded / theoretical_speedup
        axes[1].plot(data['ctx'], speedup_gap_unexpanded, label='Speedup Gap (Unexpanded)', color='green', marker='o')
    axes[1].set_title('Speedup Gap')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Context Size')
    axes[1].set_ylabel('Speedup Gap')
    axes[1].legend(loc='upper left')
    fig.savefig(image_name)
    fig.show()
    return image_name


def plot_component_relative(data, component, args):

    data.to_csv(f'benchmarks/{component}_relative_perf.csv')

    head_size = args.head_size
    p = args.p
    batch_size = args.batch_size
    num_heads = args.num_heads
    gating = args.gating
    critical_length = args.critical_length
    direction = args.direction
    from power_attention.chunk_state import ExpandedDim
    D = ExpandedDim(head_size, p)
    fig, axes = plt.subplots(2, 1, figsize=(14,20))
    name = 'Query States' if component == 'query_state' else 'Chunk States' if component == 'chunk_state' else None
    actual_speedup = data['Power Attention'] / data[name]
    theoretical_speedup = data['ctx'] / D
    speedup_gap = 1 - actual_speedup / theoretical_speedup

    axes[0].plot(data['ctx'], actual_speedup, label=f'Power Attention Runtime / {name} Runtime', color='blue', marker='o')
    axes[0].plot(data['ctx'], theoretical_speedup, label=f'Predicted Speedup', color='green', marker='o', linestyle='--')
    axes[1].plot(data['ctx'], speedup_gap, label='Speedup Gap', color='blue', marker='o')

    if args.compare_expand:
        actual_speedup_unexpanded = data[f'Power Attention-unexpanded'] / data[f'{name}-unexpanded']
        speedup_gap_unexpanded = 1 - actual_speedup_unexpanded / theoretical_speedup
        axes[0].plot(data['ctx'], actual_speedup_unexpanded, label=f'Power Attention Runtime / {name} Runtime (Unexpanded)', color='green', marker='o')
        axes[1].plot(data['ctx'], speedup_gap_unexpanded, label='Speedup Gap (Unexpanded)', color='green', marker='o')

    axes[0].set_title(
        f'{name} Relative to Power Attention-{direction} \n'
        f'(p: {p}, head_size: {head_size}, batch_size: {batch_size}, num_heads: {num_heads}, gating: {gating}, critical_length: {critical_length})',
    )
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Context Size')
    axes[0].set_ylabel('Performance ratio')
    axes[0].axvline(x=D, color='black', linestyle='--')
    axes[0].text(D, 0, f'D = {D}', color='black', ha='right', va='bottom')
    axes[0].axvline(x=2*D, color='black', linestyle='--')
    plt.text(2*D, 0, f'2D = {2*D}', color='black', ha='right', va='bottom')
    axes[0].axhline(y=1, color='red', linestyle='--')
    axes[0].text(axes[0].get_xlim()[0], 1, 'y = 1', color='red', va='bottom', ha='left')
    axes[0].axhline(y=2, color='green', linestyle='--')
    axes[0].text(axes[0].get_xlim()[0], 2, 'y = 2', color='green', va='bottom', ha='left')
    axes[0].legend(loc='upper left')

    axes[1].set_title('Speedup Gap')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Context Size')
    axes[1].set_ylabel('Speedup Gap')
    axes[1].legend(loc='upper left')
    image_name = (f'benchmarks/{component}_relative_{direction}_p_{p}_head_size_{head_size}_batch_size_{batch_size}_num_heads_{num_heads}_gating_{gating}_critical_length_{critical_length}.png')
    fig.savefig(image_name)
    fig.show()
    return image_name

def plot_power_recurrent_components(data, args):
    batch_size = args.batch_size
    num_heads = args.num_heads
    gating = args.gating
    critical_length = args.critical_length
    direction = args.direction
    chunk_size = args.chunk_size
    head_size = args.head_size
    p = args.p
    plt.figure(figsize=(10, 8))
    plt.plot(data['ctx'], data['Accumulation'], label='Accumulation', color='red', marker='o')
    plt.plot(data['ctx'], data['Chunk States'], label='Chunk States', color='blue', marker='o')
    plt.plot(data['ctx'], data['Query States'], label='Query States', color='green', marker='o')
    plt.plot(data['ctx'], data['Chunk Attention'], label='Chunk Attention', color='purple', marker='o')
    plt.plot(data['ctx'], data['Power Recurrent'], label='Power Recurrent', color='black', marker='o')
    plt.title(
        f'Power Recurrent Components-{direction} \n'
        f'(Chunk: {chunk_size}, p: {p}, head_size: {head_size}, '
        f'batch_size: {batch_size}, num_heads: {num_heads}, gating: {gating}, critical_length: {critical_length})',
    )
    plt.xscale('log')
    plt.xlabel('Context Size')
    plt.ylabel('Performance (s)')
    plt.legend(loc='upper left')
    image_name = (f'benchmarks/power_recurrent_components_{direction}_chunk_{chunk_size}_p_{p}_head_size_'
                  f'{head_size}_batch_size_{batch_size}_num_heads_{num_heads}_gating_{gating}_critical_length_{critical_length}.png')
    plt.savefig(image_name)
    plt.show()
    return image_name


def absolute_perf_benchmark(args): # noqa: C901
    p = args.p
    head_size = args.head_size
    chunk_size = args.chunk_size
    batch_size = args.batch_size
    num_heads = args.num_heads
    gating = args.gating
    critical_length = args.critical_length
    direction = args.direction

    for p in [2]:
        for head_size in [64]:
            D = math.comb(head_size + p - 1, p)
            for chunk_size in [1024]:
                res = defaultdict(list)
                for ctx in np.logspace(10, 20, num=11, base=2, dtype=np.int32).tolist():
                    try:
                        if mem_fit(ctx, chunk_size, D, head_size, batch_size, num_heads):
                            cfg = Config([batch_size, ctx, num_heads, head_size], chunk_size, p, float(D), torch.float16, torch.float32, 1e-6, critical_length=critical_length, gating=gating)
                            print('config: ', cfg.__dict__)
                            profile_whole(res, cfg, bwd=direction == 'bwd', verbose=True)
                    except torch.cuda.OutOfMemoryError:
                        print('Out of memory for config: ', cfg.__dict__)
                        break

                data = pd.DataFrame(res)
                current_date = datetime.now().date()
                
                data.to_csv(f'benchmarks/absolute_perf_benchmark_{direction}_chunk_{chunk_size}_p_{p}_head_size_{head_size}_batch_size_{batch_size}_num_heads_{num_heads}_gating_{gating}_critical_length_{critical_length}_{current_date}.csv')
                print('saved absolute perf data to ', f'benchmarks/absolute_perf_benchmark_{direction}_chunk_{chunk_size}_p_{p}_head_size_{head_size}_batch_size_{batch_size}_num_heads_{num_heads}_gating_{gating}_critical_length_{critical_length}_{current_date}.csv')
                return plot_absolute_perfs(data, direction, chunk_size, p, head_size, batch_size, num_heads, gating, critical_length, current_date)


def relative_perf_benchmark(args): # noqa: C901
    mem_limit = 80 * 1024 * 1024 * 1024  # 80 GB
    batch_size = args.batch_size
    num_heads = args.num_heads
    gating = args.gating
    critical_length = args.critical_length
    verbose = args.verbose
    compile = args.compile
    token_count = args.token_count
    direction = args.direction
    p = args.p
    head_size = args.head_size
    chunk_size = args.chunk_size
    D = math.comb(head_size + p - 1, p)
    res = defaultdict(list)
    bs_seqlen_vals = [(token_count // seqlen, seqlen) for seqlen in [2**i for i in range(10, np.log2(token_count).astype(np.int32).item())]]

    for batch_size, ctx in bs_seqlen_vals:
        try:
            res['ctx'].append(ctx)
            cfg = Config([batch_size, ctx, num_heads, head_size], chunk_size, p, float(D), torch.float16, torch.float32, 1e-6, critical_length=critical_length, gating=gating)
            print('config: ', cfg.__dict__)

            if mem_fit(ctx, chunk_size, D, head_size, batch_size, num_heads):
                profile_components(res, cfg, 'power', args)
                if args.compare_expand:
                    cfg.expand = False
                    profile_components(res, cfg, 'power', args)
                    cfg.expand = True
                profile_components(res, cfg, 'power_attn', args)
                profile_components(res, cfg, 'flash', args)
                profile_components(res, cfg, 'sdpa', args)
        
        except torch.cuda.OutOfMemoryError:
            print('Out of memory for config: ', cfg.__dict__)
            break

    data = pd.DataFrame(res)

    data.to_csv(f'benchmarks/relative_perf_benchmark_{direction}_chunk_{chunk_size}_p_{p}_head_size_{head_size}_batch_size_{batch_size}_num_heads_{num_heads}_gating_{gating}_critical_length_{critical_length}.csv')
    print('saved relative perf data to ', f'benchmarks/relative_perf_benchmark_{direction}_chunk_{chunk_size}_p_{p}_head_size_{head_size}_batch_size_{batch_size}_num_heads_{num_heads}_gating_{gating}_critical_length_{critical_length}.csv')
    relative_image = plot_relative_perfs(data, args)
    component_image = plot_power_recurrent_components(data, args)
    return relative_image, component_image


def component_relative(component, args): # noqa: C901
    mem_limit = 80 * 1024 * 1024 * 1024  # 80 GB
    p = args.p
    head_size = args.head_size
    batch_size = args.batch_size
    token_count = args.token_count
    num_heads = args.num_heads
    gating = args.gating
    critical_length = args.critical_length
    verbose = args.verbose
    direction = args.direction

    D = math.comb(head_size + p - 1, p)
    res = defaultdict(list)
    bs_seqlen_vals = [(token_count // seqlen, seqlen) for seqlen in [2**i for i in range(10, np.log2(token_count).astype(np.int32).item())]]
    for batch_size, ctx in bs_seqlen_vals:
        chunk_size = ctx
        try:
            res['ctx'].append(ctx)
            cfg = Config([batch_size, ctx, num_heads, head_size], chunk_size, p, float(D), torch.float16, torch.float32, 1e-6, critical_length=critical_length, gating=gating)
            print('config: ', cfg.__dict__)

            if mem_fit(ctx, chunk_size, D, head_size, batch_size, num_heads):
                profile_components(res, cfg, f'{component}_compare', args)
                if args.compare_expand:
                    cfg.expand = False
                    profile_components(res, cfg, f'{component}_compare', args)
                    cfg.expand = True
        
        except torch.cuda.OutOfMemoryError:
            print('Out of memory for config: ', cfg.__dict__)
            break

    data = pd.DataFrame(res)

    data.to_csv(f'benchmarks/{component}_relative_{direction}_chunk_{chunk_size}_p_{p}_head_size_{head_size}_batch_size_{batch_size}_num_heads_{num_heads}_gating_{gating}_critical_length_{critical_length}.csv')
    print('saved to ', f'benchmarks/{component}_relative_{direction}_chunk_{chunk_size}_p_{p}_head_size_{head_size}_batch_size_{batch_size}_num_heads_{num_heads}_gating_{gating}_critical_length_{critical_length}.csv')
    return plot_component_relative(data, component, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('benchmark', type=str, default='query_state_relative', help='benchmark type: relative_perf, absolute_perf, chunk_state, query_state')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--token_count', type=int, default=131072)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--p', type=int, default=2)
    parser.add_argument('--chunk_size', type=int, default=1024)
    parser.add_argument('--head_size', type=int, default=64)
    parser.add_argument('--gating', action='store_true')
    parser.add_argument('--critical_length', type=int, default=None)
    parser.add_argument('--compile', action='store_true', default=False)
    parser.add_argument('--direction', type=str, default='fwd', help='direction: fwd, bwd')
    parser.add_argument('--compare_expand', action='store_true', default=False)
    args = parser.parse_args()
    if args.benchmark == 'relative_perf':
        relative_img, component_img = relative_perf_benchmark(args)
        print('##########################################################')
        print('saved relative perf image to ', relative_img)
        print('saved component relative image to ', component_img)
    elif args.benchmark == 'absolute_perf':
        img = absolute_perf_benchmark('fwd', args)
        print('##########################################################')
        print('saved absolute perf image to ', img)
    elif args.benchmark == 'query_state':
        img = component_relative('query_state', args)
        print('##########################################################')
        print('saved query_state relative image to ', img)
    elif args.benchmark == 'chunk_state':
        img = component_relative('chunk_state', args)
        print('##########################################################')
        print('saved chunk_state relative image to ', img)
