## DISCOUNTED CUMULATIVE SUM FORWARD PASS
# This is a custom op that computes the discounted cumulative sum of a tensor.
# It is used in recurrent attention kernels to accumulate the state.
# This file contains the implementation of the forward pass as well as a fake
# implementation for tracing and testing.

## IMPLEMENTATION ##
import torch
from state_kernel_cuda import discumsum as discumsum_cuda

@torch.library.custom_op('state_kernel::discumsum_fwd', mutates_args=(), device_types='cuda')
def discumsum_fwd(X : torch.Tensor, log_G : torch.Tensor) -> torch.Tensor:
    """Compute discounted cumulative sum along time axis.

    Computes the discounted cumulative sum of X using gating factors exp(log_G).
    The time axis is the second dimension (axis=1).
    
    Input shapes:
        X: [batch, time, heads, features] - Input tensor to accumulate
        log_G: [batch, time, heads] - Log discount factors, broadcasted along features

    Output shape: 
        [batch, time+1, heads, features] - Accumulated sums
        NOTE: Output has one more timestep than input, with zeros at t=0

    Shape Restrictions:
        - Time dimension must be a multiple of 4
        - Feature dimension must be a multiple of 8
    
    The 0th (batch) and 2nd (heads) dimensions are treated as batch dimensions.
    The discount factors are broadcasted along the final feature dimension.

    TODO(jbuckman): Accept an initial state
    """
    b, n, h, d = X.shape
    cum_X = torch.empty(b, n+1, h, d, device=X.device, dtype=X.dtype)
    cum_X[:,0] = 0.
    discumsum_cuda(X, log_G, cum_X)
    return cum_X

# Fake implementation for tracing and testing
@discumsum_fwd.register_fake
def discumsum_fwd_fake(X, log_G):
    b, n, h, d = X.shape
    return torch.empty(b, n+1, h, d, device=X.device, dtype=X.dtype)

# Useful function to create sample inputs
def create_inputs(b=2, n=4, h=8, d=16, X_dtype=torch.float16, device='cuda'):
    generator = torch.Generator(device=device).manual_seed(42)
    X = torch.randn(size=(b, n, h, d), dtype=X_dtype, device=device, generator=generator)
    log_G = torch.zeros(size=(b, n, h), dtype=torch.float32, device=device) - .01
    return X, log_G


## TUTORIAL ##
if __name__ == '__main__':
    # Hyperparameters
    b, n, h, d = (2, 4, 8, 16)
    X_dtype = torch.float16
    # Create inputs
    X, log_G = create_inputs(b, n, h, d, X_dtype, 'cuda')
    # Run function
    with torch.no_grad():
        cum_X = discumsum_fwd(X, log_G)
    # Compile function, fullgraph=True confirms no graph breaks
    compiled_discumsum_fwd = torch.compile(discumsum_fwd, fullgraph=True)
    with torch.no_grad():
        for _ in range(3):
            cum_X = compiled_discumsum_fwd(X, log_G)
