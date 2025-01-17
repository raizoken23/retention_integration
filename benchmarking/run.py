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


def make_serializable(obj):
    """Convert non-serializable objects to strings."""
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(x) for x in obj]
    return obj


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