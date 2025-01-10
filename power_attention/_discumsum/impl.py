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
from power_attention._discumsum.fwd import discumsum_fwd
from power_attention._discumsum.bwd import discumsum_bwd

@torch.library.custom_op("power_attention::discumsum", mutates_args=())
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
    "power_attention::discumsum", discumsum_backward, setup_context=discumsum_setup
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
    return dict(
        X=X,
        log_G=log_G
    )


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

