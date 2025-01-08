## DISCOUNTED CUMSUM CUSTOM OP
# Computes the discounted cumsum forward pass and backward pass using CUDA.

## IMPLEMENTATION ##
import torch
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from torch.utils._pytree import tree_map
from types import NoneType
from typing import Optional, Tuple
from state_kernel._discumsum.fwd import discumsum_fwd
from state_kernel._discumsum.bwd import discumsum_bwd

@torch.library.custom_op("state_kernel::discumsum", mutates_args=())
def discumsum(X : torch.Tensor, log_G : torch.Tensor) -> torch.Tensor:
    """Compute discounted cumulative sum along time axis.

    Computes the discounted cumulative sum of X using gating factors exp(log_G).
    The time axis is the second dimension (axis=1).

    It implements the following mathematical operation:
        cum_X[t] = X[t] + exp(log_G[t]) * cum_X[t-1]

    Input shapes:
        X: [batch, time, heads, *features] - Input tensor to accumulate
        log_G: [batch, time, heads] - Log discount factors, broadcasted along features

    Output shape: 
        [batch, time+1, heads, *features] - Accumulated sums
        NOTE: Output has one more timestep than input, with zeros at t=0

    Shape Restrictions:
        - Time dimension must be a multiple of 4
        - Product of feature dimensions must be a multiple of 8
    
    The 0th (batch) and 2nd (heads) dimensions are treated as batch dimensions.
    The discount factors are broadcasted along the final feature dimensions.

    TODO(jbuckman): Accept an initial state
    """
    b, n, h, *ds = X.shape
    if len(X.shape) > 4:
        X = X.view(*X.shape[:3], -1)
    cum_X = discumsum_fwd(X, log_G)
    return cum_X.view(b, n+1, h, *ds)

# Make it traceable
@discumsum.register_fake
def discumsum_fake(X, log_G):
    b, n, h, *ds = X.shape
    return torch.empty(b, n+1, h, *ds, device=X.device, dtype=X.dtype)

# Autograd setup
def discumsum_setup(ctx, inputs, output):
    X, log_G = inputs
    ctx.save_for_backward(log_G, output)
    ctx.original_shape = X.shape

def discumsum_backward(ctx, dcum_X):
    log_G, cum_X = ctx.saved_tensors
    b, n, h, *ds = ctx.original_shape
    dcum_X = dcum_X.view(b, n+1, h, -1)
    cum_X = cum_X.view(b, n+1, h, -1)
    dX, dlog_G = discumsum_bwd(dcum_X, cum_X, log_G)
    return dX.view(b, n, h, *ds), dlog_G

# Register autograd
torch.library.register_autograd(
    "state_kernel::discumsum", discumsum_backward, setup_context=discumsum_setup
)

# Useful function to create sample inputs   
def create_inputs(b=2, n=4, h=8, D=64, d=16, X_dtype=torch.float16, device='cuda', requires_grad=False):
    generator = torch.Generator(device=device).manual_seed(42)
    X = torch.randn(size=(b, n, h, D, d), dtype=X_dtype, device=device, generator=generator)
    log_G = torch.zeros(size=(b, n, h), dtype=torch.float32, device=device) - 0.01
    if requires_grad:
        X, log_G = tree_map(
            lambda x: x.requires_grad_(True), (X, log_G)
        )
    return X, log_G


## TUTORIAL ##
if __name__ == '__main__':
    # Hyperparameters
    b, n, h, D, d = (2, 4, 8, 64, 16)
    dtype = torch.float16
    # Create inputs
    X, log_G = create_inputs(b, n, h, D, d, dtype, 'cuda', requires_grad=True)
    # Run forward
    cum_X = discumsum(X, log_G)
    # Run backward
    torch.autograd.backward(cum_X, cum_X)
    # Compile function, fullgraph=True confirms no graph breaks
    compiled_discumsum = torch.compile(discumsum, fullgraph=True)
    for _ in range(3):
        inputs = create_inputs(b, n, h, D, d, dtype, 'cuda', requires_grad=True)
        outputs = compiled_discumsum(*inputs)
        torch.autograd.backward(outputs, outputs)

## TESTS ##
import pytest
from state_kernel.checks import (
    check_tensor_property_pairs,
    check_fn_compiles,
    check_fn_compiles_with_backward,
    check_fake_fn_implementation_matches
)

TEST_CASES = [
    (2, 8, 4, 4, 32, torch.float16, 'cuda'),    # Common case with float16
    (4, 32, 8, 4, 32, torch.float16, 'cuda'),   # Large case with float16 and gating
    (1, 8, 4, 4, 64, torch.float16, 'cuda'),    # Test float16 with large head
    (2, 32, 8, 4, 64, torch.float16, 'cuda'),   # Large case with float16 and gating
    (1, 8, 4, 4, 32, torch.bfloat16, 'cuda'),   # Test bfloat16
    (4, 32, 8, 4, 32, torch.bfloat16, 'cuda'),  # Large case with bfloat16 and gating
    (2, 8, 4, 4, 64, torch.bfloat16, 'cuda'),   # Test bfloat16 with large head
    (1, 32, 8, 4, 64, torch.bfloat16, 'cuda'),  # Large case with bfloat16 and gating
]

@pytest.mark.parametrize("b,n,h,D,d,dtype,device", TEST_CASES)
def test_discumsum_create_inputs(b, n, h, D, d, dtype, device):
    X, log_G = create_inputs(b, n, h, D, d, dtype, device)
    check_tensor_property_pairs(
        (X, ((b, n, h, D, d), dtype, device)),
        (log_G, ((b, n, h), torch.float32, device))
    )

@pytest.mark.parametrize("b,n,h,D,d,dtype,device", TEST_CASES)
def test_discumsum_output(b, n, h, D, d, dtype, device):
    inputs = create_inputs(b, n, h, D, d, dtype, device)
    with torch.no_grad():
        cum_X = discumsum(*inputs)
    check_tensor_property_pairs(
        (cum_X, ((b, n+1, h, D, d), dtype, device))
    )

@pytest.mark.parametrize("b,n,h,D,d,dtype,device", TEST_CASES)
def test_discumsum_backward(b, n, h, D, d, dtype, device):
    X, log_G = create_inputs(b, n, h, D, d, dtype, device, requires_grad=True)
    output = discumsum(X, log_G)
    torch.autograd.backward(output, output)
    check_tensor_property_pairs(
        (X.grad, ((b, n, h, D, d), dtype, device)),
        (log_G.grad, ((b, n, h), torch.float32, device))
    )

@pytest.mark.parametrize("args", TEST_CASES)
def test_discumsum_compiles(args):
    check_fn_compiles(discumsum, create_inputs(*args))

@pytest.mark.parametrize("args", TEST_CASES)
def test_discumsum_compiles_with_backward(args):
    check_fn_compiles_with_backward(discumsum, create_inputs(*args, requires_grad=True))

@pytest.mark.parametrize("args", TEST_CASES)
def test_discumsum_fake_fn_implementation_matches(args):
    check_fake_fn_implementation_matches(discumsum, discumsum_fake, create_inputs(*args))

@pytest.mark.parametrize("args", TEST_CASES)
def test_discumsum_opcheck(args):
    torch.library.opcheck(discumsum, create_inputs(*args, requires_grad=True))
