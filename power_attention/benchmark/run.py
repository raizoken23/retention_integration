# Benchmarking either precision or timing of target implementations, to
# detect regressions over time.
# The idea is to run this every commit, which will produce a YAML file with the results,
# in which we can see the error and time for each test case. Error and times are nothing special,
# a benchmark could produce any scalar value.

import torch
import itertools
import click
import yaml
import inspect
import sys
from pathlib import Path
from power_attention.power_full import (
    power_full,
    power_full_reference,
    create_inputs,
)
from power_attention.checks import measure_fn_error
from power_attention.timing_utils import (
    get_compiled_versions,
    estimate_runtime,
)

### Benchmarking functions

def benchmark_power_full_error():
    """Benchmark error between target and reference implementations across different settings."""
    param_ranges = {
        'b': [1, 2],
        't': [512, 1024], 
        'h': [4, 8],
        'd': [32, 64],
        'qhead_ratio': [1, 2],
        'dtype': [torch.bfloat16],
        'device': ['cuda'],
        'gating': [False, True],
        'chunk_size': [None, 128],
        'deg': [1, 2],
        'log_space': [False, True],
    }

    results = []
    
    # Generate all combinations
    keys = param_ranges.keys()
    for values in itertools.product(*param_ranges.values()):
        params = dict(zip(keys, values))
        
        test_inputs = create_inputs(**params)
        
        # Run both implementations
        with torch.no_grad():
            max_error = measure_fn_error(power_full_reference, power_full, test_inputs)
        
        # Store results with parameters in a flat structure
        result = {**params}  # Start with all parameters
        result['value'] = max_error  # Add the measurement
        results.append(result)
    
    return results

def benchmark_power_full_timing():
    """Benchmark timing of forward and backward passes across different settings."""
    param_ranges = {
        'b': [1, 2],
        't': [512, 1024], 
        'h': [4, 8],
        'd': [32, 64],
        'qhead_ratio': [1, 2],
        'dtype': [torch.bfloat16],
        'device': ['cuda'],
        'gating': [False, True],
        'chunk_size': [None, 128],
        'deg': [1, 2],
        'log_space': [False, True],
    }

    results = []
    
    # Generate all combinations
    keys = param_ranges.keys()
    for values in itertools.product(*param_ranges.values()):
        params = dict(zip(keys, values))
        
        # Create inputs and get compiled versions
        test_inputs = create_inputs(**params, requires_grad=True)
        input_args = list(test_inputs.values())
        
        def fn(*args):
            return power_full(**dict(zip(test_inputs.keys(), args)))
            
        fwd_fn, bwd_fn, _ = get_compiled_versions(fn, *input_args)
        
        # Measure timings and create a result for each stage
        base_result = {**params}  # Start with all parameters
        
        # Forward pass timing
        fwd_result = {**base_result}
        fwd_result['stage'] = 'fwd'
        fwd_result['value'] = estimate_runtime(fwd_fn)
        results.append(fwd_result)
        
        # Backward pass timing
        bwd_result = {**base_result}
        bwd_result['stage'] = 'bwd'
        bwd_result['value'] = estimate_runtime(bwd_fn)
        results.append(bwd_result)
    
    return results

### Benchmarking machinery

def make_serializable(obj):
    """Convert non-serializable objects to strings."""
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(x) for x in obj]
    return obj

def discover_benchmarks():
    """Discover all functions starting with 'benchmark_' in current module."""
    return {
        name[10:]: obj  # Strip 'benchmark_' prefix for the key
        for name, obj in inspect.getmembers(sys.modules[__name__])
        if inspect.isfunction(obj) and name.startswith('benchmark_')
    }

def load_existing_results(output_path):
    """Load existing results from YAML file if it exists."""
    try:
        with open(output_path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}

@click.command()
@click.option('--output', '-o', 
              type=click.Path(dir_okay=False, path_type=Path),
              default=None,
              help='Output YAML file path (defaults to benchmark_results.yaml)')
@click.option('--benchmarks', '-b', 
              multiple=True,
              help='Specific benchmarks to run (without benchmark_ prefix). If none specified, runs all.')
@click.option('--reset', is_flag=True,
              help='Start with fresh results, ignoring any existing ones in the output file.')
def main(output, benchmarks, reset):
    """Run benchmarks and save results to YAML.
    
    If no benchmarks specified, runs all discovered benchmarks.
    If no output specified, saves to benchmark_results.yaml
    By default, updates existing results in the output file unless --reset is specified.
    """
    all_benchmarks = discover_benchmarks()
    
    if not all_benchmarks:
        click.echo("No benchmarks found! Add functions starting with 'benchmark_'")
        return

    # Filter benchmarks if specified
    if benchmarks:
        selected_benchmarks = {
            name: fn for name, fn in all_benchmarks.items()
            if name in benchmarks
        }
        if not selected_benchmarks:
            available = ", ".join(all_benchmarks.keys())
            click.echo(f"No matching benchmarks found. Available benchmarks: {available}")
            return
        benchmarks_to_run = selected_benchmarks
    else:
        benchmarks_to_run = all_benchmarks

    # Set default output if none specified
    if output is None:
        output = Path('./benchmark_results.yaml')

    # Load existing results unless reset is specified
    results = {} if reset else load_existing_results(output)
    
    # Run selected benchmarks and collect results
    with click.progressbar(benchmarks_to_run.items(), label='Running benchmarks') as bar:
        for name, benchmark_fn in bar:
            results[name] = benchmark_fn()
    
    # Make results serializable before saving
    serializable_results = make_serializable(results)
    
    # Save results to YAML
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        yaml.safe_dump(serializable_results, f, default_flow_style=False, sort_keys=False)
    
    click.echo(f"Benchmark results {'saved to' if reset else 'updated in'} {output}")

if __name__ == '__main__':
    main() 