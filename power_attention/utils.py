import inspect
import os
from math import floor, ceil
from collections import defaultdict
from contextlib import contextmanager
from functools import partial, wraps

# Optional dependencies
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
from torch.utils._pytree import tree_map
from torch.autograd.profiler import record_function
from power_attention.checks import clone_grads, create_random_grads, sanity_check

DEFAULT_SEEDS = [40 + i for i in range(2)]
class DummyCtx:
    def save_for_backward(self, *args):
        self.saved_tensors = args

def dummify(fn):
    return partial(fn, DummyCtx())

@contextmanager
def record_ctx(name):
    if not torch.compiler.is_dynamo_compiling() and os.environ.get('POWER_PROFILE'):
        with record_function(name):
            yield
            torch.cuda.synchronize()
    else:
        yield

def record_fn(name):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            if not torch.compiler.is_dynamo_compiling() and os.environ.get('POWER_PROFILE'):
                with record_function(name):
                    res = fn(*args, **kwargs)
                    torch.cuda.synchronize()
                    return res
            else:
                return fn(*args, **kwargs)
        return wrapper
    return decorator

@contextmanager
def no_record():
    old_profile = os.environ.get('POWER_PROFILE')
    if old_profile:
        del os.environ['POWER_PROFILE']
    yield
    if old_profile:
        os.environ['POWER_PROFILE'] = old_profile

def partial_with_keywords(func, /, **fixed_kwargs):
    """Enables calls like
    >>> my_func = lambda a, b, c: a + b + c
    >>> partial_with_keywords(my_func, b=2)(1, c=3)
    6
    
    Creates a partial function that fixes some keyword arguments while allowing
    the remaining arguments to be passed either positionally or by keyword.
    
    Args:
        func: The function to create a partial from
        fixed_kwargs: Keyword arguments to fix in the partial function
        
    Returns:
        A wrapped function that combines fixed kwargs with passed arguments
    """
    sig = inspect.signature(func)
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Validate fixed kwargs against signature
        for k in fixed_kwargs:
            if k not in sig.parameters:
                raise TypeError(f"{func.__name__}() got an unexpected keyword argument '{k}'")
        # Convert positional args to kwargs by matching with signature params
        param_names = list(sig.parameters.keys())
        args_as_kwargs = {}
        fixed_param_names = set(fixed_kwargs.keys())
        arg_idx = 0
        for param_name in param_names:
            if param_name in fixed_param_names:
                continue
            if arg_idx < len(args):
                args_as_kwargs[param_name] = args[arg_idx]
                arg_idx += 1
        # Combine all kwargs, with passed kwargs taking highest precedence
        combined_kwargs = {**fixed_kwargs, **args_as_kwargs, **kwargs}
        try:
            return func(**combined_kwargs)
        except TypeError as e:
            raise TypeError(f"Error calling {func.__name__}: {str(e)}") from None
    return wrapper

def register_multi_grad_hook(tensors, fn):
    """ Register a hook that will only call fn when all tensor gradients are available.
    """
    count = 0
    nb_calls = None
    buffer = None

    def get_grad_fn(t):
        # or grad accumulator
        if t.requires_grad and t.grad_fn is None:
            return t.clone().grad_fn.next_functions[0][0]
        else:
            return t.grad_fn

    grad_fns = list(map(get_grad_fn, tensors))

    def get_inner_hook(idx):
        def inner_hook(grad):
            nonlocal count, nb_calls, buffer

            if count == 0:
                # On the first call, compute the actual nb_calls and buffer
                nb_calls = sum(1 for g in grad_fns if torch._C._will_engine_execute_node(g))
                buffer = [None] * nb_calls

            buffer[idx] = grad
            count += 1

            if count == nb_calls:
                fn(buffer)
        return inner_hook
    for i, t in enumerate(tensors):
        t.register_hook(get_inner_hook(i))

def register_multi_grad_hook(tensors, fn):
    """ Register a hook that will only call fn when all tensor gradients are available.
    """
    count = 0
    nb_calls = None
    buffer = None

    def get_grad_fn(t):
        # or grad accumulator
        if t.requires_grad and t.grad_fn is None:
            return t.clone().grad_fn.next_functions[0][0]
        else:
            return t.grad_fn

    grad_fns = list(map(get_grad_fn, tensors))

    def get_inner_hook(idx):
        def inner_hook(grad):
            nonlocal count, nb_calls, buffer

            if count == 0:
                # On the first call, compute the actual nb_calls and buffer
                nb_calls = sum(1 for g in grad_fns if torch._C._will_engine_execute_node(g))
                buffer = [None] * nb_calls

            buffer[idx] = grad
            count += 1

            if count == nb_calls:
                fn(buffer)
        return inner_hook
    for i, t in enumerate(tensors):
        t.register_hook(get_inner_hook(i))

# Credit: https://github.com/pytorch/pytorch/issues/64947#issuecomment-2304371451
def torch_quantile(
    input: torch.Tensor,
    q: float | torch.Tensor,
    dim: int | None = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Sanitization: dim
    # Because one cannot pass  `dim=None` to `squeeze()` or `kthvalue()`
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Sanitization: inteporlation
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Sanitization: out
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Logic
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Rectification: keepdim
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)

    return out



def get_statistics(t):
    q25 = torch_quantile(t, 0.25).item()
    q50 = torch_quantile(t, 0.5).item()
    q75 = torch_quantile(t, 0.75).item()
    q99 = torch_quantile(t, 0.99).item()
    q999 = torch_quantile(t, 0.999).item()
    q9999 = torch_quantile(t, 0.9999).item()
    q10000 = torch_quantile(t, 1.0).item()
    return {
        # 'q25': q25,
        'q50': q50,
        'q75': q75,
        # 'q99': q99,
        'q999': q999,
        # 'q9999': q9999,
        # 'q10000': q10000,
    }

def get_comprehensive_stats(t, error):
    error_stats = get_statistics(error)
    return {
        # 'mean': t.mean().item(),
        # 'abs_mean': t.abs().mean().item(),
        'std': t.std().item(),
        # 'abs_std': t.abs().std().item(),
        # 'min': t.min().item(),
        'max': t.max().item(),
        **error_stats
    }

def get_precision(*, gold_fn, gold_input_factory, ref_fn, ref_input_factory, test_fn, test_input_factory, precision_threshold=1e-4, seeds=None):
    ref_precisions = defaultdict(list)
    test_precisions = defaultdict(list)
    for seed in seeds if seeds is not None else DEFAULT_SEEDS:
        gold_output = gold_fn(*gold_input_factory(seed))
        ref_output = ref_fn(*ref_input_factory(seed))
        test_output = test_fn(*test_input_factory(seed))
        if hasattr(gold_output, '__iter__'):
            assert len(gold_output) == len(ref_output) == len(test_output), "gold, ref, and test must have the same length"
        for i, (gold, ref, test) in enumerate(zip(gold_output, ref_output, test_output)):
            ref_abs_error = ((gold - ref).abs())
            test_abs_error = ((gold - test).abs())
            ref_precisions[i].append(get_statistics(ref_abs_error))
            test_precisions[i].append(get_statistics(test_abs_error))
    # calculate max precision
    rep_stats, test_stats = {}, {}
    for i in range(len(ref_precisions)):
        rep_stats[i] = {stats: np.mean([v[stats] for v in ref_precisions[i]]) for stats in ref_precisions[i][0].keys()}
        test_stats[i] = {stats: np.mean([v[stats] for v in test_precisions[i]]) for stats in test_precisions[i][0].keys()}
    return rep_stats, test_stats


def get_precision_bwd(*, gold_fn, gold_input_factory, ref_fn, ref_input_factory, test_fn, test_input_factory, precision_threshold=1e-4, grad_scale=1e-4, seeds=None, create_grads_fn=None):
    ref_precisions = defaultdict(list)
    test_precisions = defaultdict(list)

    def make_bwd(fn, input_factory, seed):
        inputs = input_factory(seed)
        for t in inputs:
            if isinstance(t, torch.Tensor):
                t.requires_grad_(True)
                t.retain_grad()
        outputs = fn(*inputs)
        def _bwd(grads):
            torch.autograd.backward(outputs, grad_tensors=grads, retain_graph=False)
            return clone_grads(inputs)
        return _bwd

    for seed in seeds if seeds is not None else DEFAULT_SEEDS:
        gold_bwd = make_bwd(gold_fn, gold_input_factory, seed)
        ref_bwd = make_bwd(ref_fn, ref_input_factory, seed)
        test_bwd = make_bwd(test_fn, test_input_factory, seed)
        
        gold_output = gold_fn(*gold_input_factory(seed))
        ref_output = ref_fn(*ref_input_factory(seed))
        test_output = test_fn(*test_input_factory(seed))
        if create_grads_fn is not None:
            grad_in_gold = create_grads_fn(gold_output, seed=seed)
            grad_in_ref = create_grads_fn(ref_output, seed=seed)
            grad_in_test = create_grads_fn(test_output, seed=seed)
        else:
            grad_in_gold = create_random_grads(gold_output, scale=grad_scale, seed=seed)
            grad_in_ref = create_random_grads(ref_output, scale=grad_scale, seed=seed)
            grad_in_test = create_random_grads(test_output, scale=grad_scale, seed=seed)
        
        gold_grads = gold_bwd(grad_in_gold)
        ref_grads = ref_bwd(grad_in_ref)
        test_grads = test_bwd(grad_in_test)

        if hasattr(gold_grads, '__iter__'):
            assert len(gold_grads) == len(ref_grads) == len(test_grads), "gold, ref, and test must have the same length"
        sanity_check(gold_grads)
        sanity_check(ref_grads)
        sanity_check(test_grads)
        for i, (gold, ref, test) in enumerate(zip(gold_grads, ref_grads, test_grads)):
            if gold is None and ref is None and test is None:
                continue
            ref_abs_error = ((gold - ref).abs())
            test_abs_error = ((gold - test).abs())
            ref_precisions[i].append(get_statistics(ref_abs_error))
            test_precisions[i].append(get_statistics(test_abs_error))

    # calculate max stats
    rep_stats, test_stats = {}, {}
    for i in range(len(ref_precisions)):
        rep_stats[i] = {stats: np.mean([v[stats] for v in ref_precisions[i]]) for stats in ref_precisions[i][0].keys()}
        test_stats[i] = {stats: np.mean([v[stats] for v in test_precisions[i]]) for stats in test_precisions[i][0].keys()}
    return rep_stats, test_stats


# this method makes it so that we don't have to use torch.autograd.backward, which requires the output gradients
# to be matching the input tensors, preventing us from extracting intermediate values
def make_fwd_bwd(fwd_fn, bwd_fn, grad_scale=1e-4, seed=42, output_selector=lambda x: x):
    def fwd_bwd(*inputs):
        ctx = DummyCtx()
        outputs = fwd_fn(ctx, *inputs)
        bwd_inputs = create_random_grads(outputs, scale=grad_scale, seed=seed)
        return output_selector(bwd_fn(ctx, *bwd_inputs))
    return fwd_bwd


def get_stats_and_precision(*, gold_fn, gold_input_factory, ref_fn, ref_input_factory, test_fn, test_input_factory, seeds=None):
    ref_stats_all = defaultdict(list)
    test_stats_all = defaultdict(list)
    for seed in seeds if seeds is not None else DEFAULT_SEEDS:
        gold_output = gold_fn(*gold_input_factory(seed))
        ref_output = ref_fn(*ref_input_factory(seed))
        test_output = test_fn(*test_input_factory(seed))
        if hasattr(gold_output, '__iter__'):
            assert len(gold_output) == len(ref_output) == len(test_output), "gold, ref, and test must have the same length"
        for i, (gold, ref, test) in enumerate(zip(gold_output, ref_output, test_output)):
            ref_abs_error = ((gold - ref).abs())
            test_abs_error = ((gold - test).abs())
            ref_stats_all[i].append(get_comprehensive_stats(ref, ref_abs_error))
            test_stats_all[i].append(get_comprehensive_stats(test, test_abs_error))
    # calculate mean stats
    rep_stats, test_stats = {}, {}
    for i in range(len(ref_stats_all)):
        rep_stats[i] = {stats: np.mean([v[stats] for v in ref_stats_all[i]]) for stats in ref_stats_all[i][0].keys()}
        test_stats[i] = {stats: np.mean([v[stats] for v in test_stats_all[i]]) for stats in test_stats_all[i][0].keys()}
    return rep_stats, test_stats


def plot_precision(ref_precisions, test_precisions, xaxis, xlabel, title, precision_threshold=1e-4, x_log=False, y_log=True, test_label='test', ref_label='reference'):
    """
    ref_precisions: dict[int, list[dict[str, float]]]
    test_precisions: dict[int, list[dict[str, float]]]
    """
    num_args = len(ref_precisions)
    fig, axs = plt.subplots(num_args, 1, figsize=(12, 10), layout="constrained")
    if num_args == 1:
        axs = [axs]
    colors = plt.cm.tab10.colors
    for i, (ref_p, test_p) in enumerate(zip(ref_precisions.values(), test_precisions.values())):
        for j, stat in enumerate(ref_p[0].keys()):    
            axs[i].plot(xaxis, [v[stat] for v in ref_p], label=f'{ref_label} {stat}', color=colors[j])
            axs[i].plot(xaxis, [v[stat] for v in test_p], label=f'{test_label} {stat}', linestyle='--', color=colors[j])
        axs[i].set_xlabel(xlabel)
        # axs[i].set_ylim(1e-9, 10e-4)
        if y_log:
            axs[i].set_yscale('log')
        if x_log:
            axs[i].set_xscale('log')
        axs[i].set_ylabel(f'Element {i} abs error stats')
        axs[i].legend()
    plt.show()
    fig.suptitle(title)
    path = f"{title.strip().replace(' ', '_').replace('=', '_').replace(',', '_')}.png".replace("'", '')
    plt.savefig(path)
    print(f"Saved to {path}")


def plot_stats(ref_stats, test_stats, xaxis, xlabel, title, precision_threshold=1e-4, x_log=False, y_log=True, test_label='test', ref_label='reference', element_names=None, only_ref=False):
    """
    ref_stats: dict[int, list[dict[str, float]]]
    test_stats: dict[int, list[dict[str, float]]]
    """
    num_args = len(ref_stats)
    num_stats = len([k for k in ref_stats[0][0].keys() if not k.startswith('q')]) + 1
    stats_keys = [k for k in ref_stats[0][0].keys() if not k.startswith('q')]
    precision_keys = [k for k in ref_stats[0][0].keys() if k.startswith('q')]
    fig, axs = plt.subplots(num_args, num_stats, figsize=(6*num_stats, 6*num_args), layout="constrained")

    if num_args == 1:
        axs = np.reshape(axs, (1, num_stats))

    colors = plt.cm.tab10.colors
    for i, (ref_p, test_p) in enumerate(zip(ref_stats.values(), test_stats.values())):
        for j, stat in enumerate(stats_keys):    
            axs[i, j].plot(xaxis, [v[stat] for v in ref_p], label=f'{ref_label} {stat}', color=colors[j])
            if not only_ref:
                axs[i, j].plot(xaxis, [v[stat] for v in test_p], label=f'{test_label} {stat}', linestyle='--', color=colors[j])
            axs[i, j].set_xlabel(xlabel)
            # axs[i, j].set_ylim(1e-9, 10e-4)
            if y_log:
                axs[i, j].set_yscale('log')
            if x_log:
                axs[i, j].set_xscale('log')
            axs[i, j].set_ylabel(f'Element {i if element_names is None else element_names[i]} {stat}')
            axs[i, j].legend()
        for k, key in enumerate(precision_keys):
            axs[i, num_stats - 1].plot(xaxis, [v[key] for v in ref_p], label=f'{ref_label} {key}', color=colors[j + k + 1])
            if not only_ref:
                axs[i, num_stats - 1].plot(xaxis, [v[key] for v in test_p], label=f'{test_label} {key}', linestyle='--', color=colors[j + k + 1])
        axs[i, num_stats - 1].set_xlabel(xlabel)
        if x_log:
            axs[i, num_stats - 1].set_xscale('log')
        if y_log:
            axs[i, num_stats - 1].set_yscale('log')
        axs[i, num_stats - 1].set_ylabel(f'Element {i if element_names is None else element_names[i]} error stats')
        axs[i, num_stats - 1].legend()
    plt.show()
    fig.suptitle(title)
    path = f"{title.strip().replace(' ', '_').replace('=', '_').replace(',', '_')}.png".replace("'", '')
    plt.savefig(path)
    print(f"Saved to {path}")
    

def print_tensor(tensor, indent=0, multi_idx=None):
    """Prints a tensor in a readable format.
    
    For 1D/2D tensors, prints in CSV-like format.
    For higher dimensions, prints recursively with headers for each slice.
    For scalar tensors, prints the value directly.
    
    Args:
        tensor: torch.Tensor to print
        indent: Number of spaces to indent (used in recursive calls)
        multi_idx: List of indices for higher dimensional tensors (used in recursive calls)
    """
    if multi_idx is None:
        multi_idx = []
    
    # Handle scalar tensors (0-dimensional)
    if tensor.dim() == 0:
        print(' ' * indent + str(tensor.item()))
        return
        
    if tensor.dim() <= 2:
        # Convert to numpy for prettier printing
        array = tensor.detach().to(torch.float32).cpu().numpy()
        # Create pandas DataFrame for nice formatting
        if tensor.dim() == 1:
            array = array.reshape(1, -1)
        df = pd.DataFrame(array)
        # Print with proper indentation
        for line in df.to_string(header=False, index=False).split('\n'):
            print(' ' * indent + line)
            
    else:
        # Handle higher dimensional tensors recursively
        for i in range(tensor.shape[0]):
            current_idx = multi_idx + [i]
            idx_str = f"[{','.join(map(str, current_idx))}]"
            print(' ' * indent + f"Index {idx_str}:")
            print_tensor(tensor[i], indent + 2, current_idx)
            if i < tensor.shape[0] - 1:
                print()
    