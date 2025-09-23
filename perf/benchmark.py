""" This script enables running things we care about
"""
import torch
import copy
import time
import argparse
from pathlib import Path
from perf._timing import benchmark_speed
from collections import defaultdict
from typing import Dict, List, Any, Callable, Iterator, Optional
from tabulate import tabulate
import os
import logging
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

from .db import KVDB
from .benchmarks.runs import *  # noqa
from ._utils import tensors_to_ones_like
from retention._update_state import default_D
from vidrial.py_utils.gpu import get_cuda_device_basic_props
from vidrial.jit.settings import settings, PickBest

benchmark_db = KVDB(os.path.expanduser('~/.retention-benchmark.db'))
logger = logging.getLogger(__name__)
plots_dir = Path(__file__).parent.parent / 'plots'

# Increase PyTorch compilation cache size to avoid recompilation
torch._dynamo.config.cache_size_limit = 512

def str_to_dtype(s: str):
    if s == 'float16':
        return torch.float16
    elif s == 'float32':
        return torch.float32
    elif s == 'bfloat16':
        return torch.bfloat16
    else:
        raise ValueError(f"Invalid dtype: {s}")


PROFILABLE_RUNS = {
    'sdpa': SDPA.make_run,
    'power_retention_triton': PowerRetentionTriton.make_run,
    'power_retention_vidrial': PowerRetentionVidrial.make_run,
    'query_state_triton': QueryStateTriton.make_run,
    'query_state_vidrial': QueryStateVidrial.make_run,
    'update_state_triton': UpdateStateTriton.make_run,
    'update_state_vidrial': UpdateStateVidrial.make_run,
    'flash_attn': FlashAttn.make_run,
    'discumsum': Discumsum.make_run,
}

class BenchmarkConfig:
    """Configuration for benchmark runs."""
    
    def __init__(self, b: int = 1, t: int = 65536, h: int = 8, d: int = 64, 
                 dtype: str = 'bfloat16', device: str = 'cuda', deg: int = 2,
                 chunk_size: int = 1024, gating: bool = True, norm: bool = True,
                 compile: bool = True, measure: bool = True):
        self.b = b
        self.t = t
        self.h = h
        self.d = d
        self.dtype = dtype
        self.device = device
        self.deg = deg
        self.chunk_size = chunk_size
        self.gating = gating
        self.norm = norm
        self.compile = compile
        self.measure = measure
        
        # Computed properties
        self.n = t // chunk_size
        self.D = default_D(d, deg)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for function calls."""
        return {
            'b': self.b,
            't': self.t,
            'c': self.t,
            'h': self.h,
            'd': self.d,
            'n': self.n,
            'dtype': self.dtype,
            'device': self.device,
            'deg': self.deg,
            'norm': self.norm,
            'D': self.D,
            'chunk_size': self.chunk_size,
            'gating': self.gating,
        }


def run_kernel(kernel: str, config: BenchmarkConfig, mode: str) -> float:
    """Run a single kernel with the given configuration and mode."""
    logger.info(f"Benchmarking {kernel} with {mode} mode")
    if kernel not in PROFILABLE_RUNS:
        raise ValueError(f"Unknown kernel: {kernel}. Available kernels: {list(PROFILABLE_RUNS.keys())}")
    
    # Prepare kwargs
    kwargs = config.to_dict()
    kwargs['requires_grad'] = 'bwd' in mode
    
    # Create cache key
    key = copy.deepcopy(kwargs) | {'mode': mode, 'compile': config.compile, 'kernel': kernel}
    
    # Check cache
    if os.environ.get('UPDATE_DB', '1') == '0' and key in benchmark_db:
        logger.info(f"Using cached result for {kernel} {mode}")
        return benchmark_db.get(lambda k: k == key)[0][1]
    
    # Convert dtype string to torch dtype
    kwargs['dtype'] = str_to_dtype(kwargs['dtype'])
    
    # Create and run the kernel
    try:
        run_fn = PROFILABLE_RUNS[kernel](**kwargs)
        logger.info(f"Running {kernel} with {mode} mode")
        
        if config.measure:
            start = time.time()
            ms = benchmark_speed(
                direction=mode,
                fn=run_fn,
                create_inputs=lambda **kw: {},
                create_inputs_kwargs={},
                compile=config.compile,
                num1=3,
                num2=10,
                warmup=1,
            )
            logger.info(f"Kernel {kernel} completed in {ms:.2f} ms (total time: {time.time() - start:.2f}s)")
            benchmark_db.put(key, ms)
            return ms
        else:
            # Run without measurement
            if mode == 'fwd':
                run_fn()
            elif 'bwd' in mode:
                outputs = run_fn()
                grads = tensors_to_ones_like(outputs)
                tuples = [(o, g) for o, g in zip(outputs, grads) if o.requires_grad]
                outputs, grads = [t[0] for t in tuples], [t[1] for t in tuples]
                torch.autograd.backward(outputs, grad_tensors=grads, retain_graph=True)
            else:
                raise ValueError(f"Invalid mode: {mode}")
            return 0.0
            
    except Exception as e:
        logger.error(f"Failed to run {kernel}: {e}")
        raise


def run_kernel_with_chunk_adjustment(kernel: str, config: BenchmarkConfig, mode: str) -> float:
    """Run kernel with automatic chunk size adjustment for certain kernels."""
    # Kernels that need chunk_size adjustment
    chunk_kernels = {
        'query_state_triton', 'query_state_vidrial', 'update_state_triton', 'update_state_vidrial',
    }
    attn_kernels = {'power_attention_triton'}

    adjusted_config = copy.deepcopy(config)
    if kernel in chunk_kernels:
        adjusted_config.t = config.chunk_size
    elif kernel in attn_kernels:
        adjusted_config.b = config.b * config.n
        adjusted_config.t = config.chunk_size
        adjusted_config.norm = False
    
    return run_kernel(kernel, adjusted_config, mode)


def benchmark_single_kernel(kernel: str, config: BenchmarkConfig, mode: str) -> None:
    """Benchmark a single kernel and print results."""
    try:
        result = run_kernel_with_chunk_adjustment(kernel, config, mode)
        if config.measure:
            print(f"Kernel: {kernel}")
            print(f"Mode: {mode}")
            print(f"Result: {result:.3f} ms")
            print(f"Configuration: b={config.b}, t={config.t}, h={config.h}, d={config.d}, "
                f"dtype={config.dtype}, deg={config.deg}, chunk_size={config.chunk_size}")
        else:
            print(f"Kernel {kernel} executed successfully (no measurement)")
    except Exception as e:
        print(f"Error running {kernel}: {e}")
        raise


def benchmark_all_kernels(config: BenchmarkConfig, mode: str, title: str = '') -> None:
    """Benchmark all kernels and create plots."""
    problem_str = f"b_{config.b}_t_{config.t}_n_{config.n}_h_{config.h}_d_{config.d}_dtype_{config.dtype}_device_{config.device}_deg_{config.deg}_chunk_size_{config.chunk_size}_gating_{config.gating}_mode_{mode}_compile_{config.compile}"
    
    # Run power full kernels
    power_retention_triton = run_kernel_with_chunk_adjustment('power_retention_triton', config, mode)
    power_retention_vidrial = run_kernel_with_chunk_adjustment('power_retention_vidrial', config, mode)
    
    # Component kernels
    query_state_triton = run_kernel_with_chunk_adjustment('query_state_triton', config, mode)
    query_state_vidrial = run_kernel_with_chunk_adjustment('query_state_vidrial', config, mode)
    update_state_triton = run_kernel_with_chunk_adjustment('update_state_triton', config, mode)
    update_state_vidrial = run_kernel_with_chunk_adjustment('update_state_vidrial', config, mode)
    discumsum = run_kernel('discumsum', config, mode)
    chunked_attention_triton = run_kernel_with_chunk_adjustment('power_attention_triton', config, mode)
    
    # Baseline kernels
    sdpa = run_kernel('sdpa', config, mode)
    flash_attn = run_kernel('flash_attn', config, mode)

    if not config.measure:
        return

    # Create plot
    data = {
        'Implementation': [
            'Power Full (Triton)', 'Power Full (Vidrial)',
            'Power Full Breakdown (Triton)', 'Power Full Breakdown (Vidrial)',
            'SDPA', 'Flash Attention',
        ],
        'Query State (ms)': [0, 0, query_state_triton, query_state_vidrial, 0, 0],
        'Update State (ms)': [0, 0, update_state_triton, update_state_vidrial, 0, 0],
        'Chunked Attention (ms)': [0, 0, chunked_attention_triton, chunked_attention_triton, 0, 0],
        'Discumsum (ms)': [0, 0, discumsum, discumsum, 0, 0],
        'Total (ms)': [
            power_retention_triton, power_retention_vidrial,
            0, 0,
            sdpa, flash_attn,
        ]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(plots_dir / f'benchmark_results_{problem_str}.csv')
    df = df.set_index('Implementation')
    ax = df.plot(kind='bar', stacked=True, figsize=(12, 7))
    plt.xticks(rotation=15, ha='right')
    gpu_name = get_cuda_device_basic_props()[0]['name'].replace(' ', '_')
    plt.ylabel('Time (milliseconds)')
    plt.title(f'Performance Comparison of Different Implementations {title}')

    # Add value labels on top of each bar
    for i in range(len(df.index)):
        total = df.iloc[i].sum()
        if total > 0:  # Only add label if bar has height
            ax.text(i, total, f'{total:.1f}ms',
                   ha='center', va='bottom')
    
    # Add problem details as text below the graph
    problem_details = f'Problem: batch={config.b}, seq_len={config.t}, heads={config.h}, head_dim={config.d}, n_chunks={config.n}, dtype={config.dtype}, deg={config.deg}, chunk_size={config.chunk_size}, gating={config.gating}, mode={mode}, compile={config.compile}'
    hardware_details = f'Hardware: {gpu_name.replace("_", " ")}'
    plt.figtext(0.5, 0.00, f'{problem_details}\n{hardware_details}', 
                ha='center', va='bottom', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)  # Make room for the text below
    plt.savefig(plots_dir / f'benchmark_results_{problem_str}_{gpu_name}_{title}.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved figure to {plots_dir / f'benchmark_results_{problem_str}_{gpu_name}_{title}.png'}")


def plot_throughput_by_ctx(base_config: BenchmarkConfig, ts: list[int], mode: str, impl: str = 'default'):
    """Plot throughput vs context length for different implementations."""
    data = defaultdict(list)
    
    for t in ts:
        config = copy.deepcopy(base_config)
        config.t = t
        config.n = t // config.chunk_size  # Recalculate n for new t
        
        data['ctx'].append(t)
        data['tokens'].append(config.b * t)
        data['power_retention_triton'].append(run_kernel('power_retention_triton', config, mode))
        data['power_retention_vidrial'].append(run_kernel('power_retention_vidrial', config, mode))
        data['sdpa'].append(run_kernel('sdpa', config, mode))
        data['flash_attn'].append(run_kernel('flash_attn', config, mode))

    df = pd.DataFrame(data)
    for kernel in ['power_retention_triton', 'power_retention_vidrial', 'sdpa', 'flash_attn']:
        df[kernel] = df['tokens'] / df[kernel]
    del df['tokens']
    
    df = df.set_index(['ctx'])
    df.plot(kind='line', figsize=(12, 7), linewidth=3, marker='o')
    plt.ylabel('Throughput (tokens/s)')
    gpu_name = get_cuda_device_basic_props()[0]['name'].replace(' ', '_')
    plt.title('Throughput vs Context Length Comparison')

    # Add problem details as text below the graph
    problem_details = f'Problem: batch={base_config.b}, heads={base_config.h}, head_dim={base_config.d}, dtype={base_config.dtype}, deg={base_config.deg}, chunk_size={base_config.chunk_size}, gating={base_config.gating}, mode={mode}, compile={base_config.compile}'
    hardware_details = f'Hardware: {gpu_name.replace("_", " ")}'
    plt.figtext(0.5, 0.02, f'{problem_details}\n{hardware_details}', 
                ha='center', va='bottom', fontsize=9, style='italic')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the text below
    fig_name = f'throughput_by_ctx_b_{base_config.b}_n_{base_config.n}_h_{base_config.h}_d_{base_config.d}_dtype_{base_config.dtype}_device_{base_config.device}_deg_{base_config.deg}_chunk_size_{base_config.chunk_size}_gating_{base_config.gating}_mode_{mode}_compile_{base_config.compile}'
    plt.savefig(plots_dir / f'{fig_name}_{gpu_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved figure to {plots_dir / f'{fig_name}_{gpu_name}.png'}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Power Attention Benchmark CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all kernels with plotting (original behavior)
  python benchmark.py fwd
  python benchmark.py bwd
  python benchmark.py fwd+bwd

  # Run individual kernels (new behavior)
  python benchmark.py query_state fwd
  python benchmark.py query_state_vidrial fwd
  python benchmark.py power_retention_vidrial bwd
  python benchmark.py sdpa fwd --b 4 --t 8192

  # List available kernels
  python benchmark.py --list-kernels
        """
    )
    
    # First argument can be either a kernel name or a mode
    parser.add_argument('first_arg', nargs='?', 
                       help='Either kernel name (for single kernel run) or mode (for all kernels)')
    parser.add_argument('second_arg', nargs='?', 
                       help='Mode when first argument is a kernel name')
    
    # Configuration arguments
    parser.add_argument('--b', type=int, default=1, help='Batch size (default: 1)')
    parser.add_argument('--t', type=int, default=65536, help='Sequence length (default: 65536)')
    parser.add_argument('--h', type=int, default=8, help='Number of heads (default: 8)')
    parser.add_argument('--d', type=int, default=64, help='Head dimension (default: 64)')
    parser.add_argument('--dtype', type=str, default='bfloat16', 
                       choices=['float16', 'float32', 'bfloat16'],
                       help='Data type (default: bfloat16)')
    parser.add_argument('--deg', type=int, default=2, help='Degree (default: 2)')
    parser.add_argument('--chunk-size', type=int, default=1024, help='Chunk size (default: 1024)')
    parser.add_argument('--title', type=str, default='', help='Title of the plot')
    parser.add_argument('--no-measure', action='store_true', help='Do not measure the performance, simply run the kernel')
    parser.add_argument('--compile', action='store_true', help='Compile the kernel')
    parser.add_argument('--list-kernels', action='store_true', help='List available kernels and exit')
    
    args = parser.parse_args()
    
    # Handle --list-kernels
    if args.list_kernels:
        print("Available kernels:")
        for kernel in sorted(PROFILABLE_RUNS.keys()):
            print(f"  {kernel}")
        return
    
    # Parse arguments to determine mode
    if args.first_arg is None:
        parser.error("Must provide either a kernel name or execution mode")
    
    # Determine if we're running a single kernel or all kernels
    modes = ['fwd', 'bwd', 'fwd+bwd']
    all_kernels = list(PROFILABLE_RUNS.keys())
    
    if args.first_arg in modes:
        # Original behavior: run all kernels with plotting
        if args.second_arg is not None:
            parser.error(f"Unexpected second argument when mode is specified: {args.second_arg}")
        
        mode = args.first_arg
        config = BenchmarkConfig(
            b=args.b,
            t=args.t,
            h=args.h,
            d=args.d,
            dtype=args.dtype,
            device='cuda',
            deg=args.deg,
            chunk_size=args.chunk_size,
            gating=True,
            norm=True,
            compile=args.compile,
            measure=not args.no_measure,
        )
        
        benchmark_all_kernels(config, mode, args.title)
        
    elif args.first_arg in all_kernels:
        # New behavior: run single kernel
        if args.second_arg is None:
            parser.error(f"Must provide execution mode after kernel name: {args.first_arg}")
        if args.second_arg not in modes:
            parser.error(f"Invalid mode: {args.second_arg}. Must be one of {modes}")
        
        kernel = args.first_arg
        mode = args.second_arg
        config = BenchmarkConfig(
            b=args.b,
            t=args.t,
            h=args.h,
            d=args.d,
            dtype=args.dtype,
            device='cuda',
            deg=args.deg,
            chunk_size=args.chunk_size,
            gating=True,
            norm=True,
            compile=args.compile,
            measure=not args.no_measure,
        )
        
        benchmark_single_kernel(kernel, config, mode)
        
    else:
        parser.error(f"Unknown kernel or mode: {args.first_arg}. Use --list-kernels to see available options.")

# Uncomment to enable throughput vs context length benchmarking
# def run_throughput_benchmark():
#     config = BenchmarkConfig(
#         b=1,
#         h=8,
#         d=64,
#         dtype='bfloat16',
#         device='cuda',
#         deg=2,
#         chunk_size=1024,
#         gating=True,
#         compile=True,
#         measure=True,
#     )
#     plot_throughput_by_ctx(
#         base_config=config,
#         ts=[4096, 8192, 16384, 32768, 65536],
#         mode='fwd',
#         impl='default',
#     )


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.DEBUG)
    # Disable matplotlib debug logging
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    with settings.set(policy=PickBest, max_workers=16, allow_failure=True, verbose=True):
        main()
