## DISCOUNTED CUMULATIVE SUM BACKWARD PASS

## IMPLEMENTATION ##
import torch
from power_attention_cuda import discumsum_bwd as discumsum_bwd_cuda

@torch.library.custom_op('power_attention::discumsum_bwd', mutates_args=(), device_types='cuda')
def discumsum_bwd(dout : torch.Tensor, out : torch.Tensor, log_G : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute gradients for discounted cumulative sum operation.

    Computes gradients with respect to inputs for the discounted cumulative sum operation.
    The time axis is the second dimension (axis=1).
    
    Input shapes:
        log_G: [batch, time, heads] - Log discount factors used in forward pass
        dout: [batch, time+1, heads, features] - Gradient of loss with respect to forward output
        out: [batch, time+1, heads, features] - Forward pass output (needed for gradient computation)

    Output shapes:
        dX: [batch, time, heads, features] - Gradient with respect to input X
        dlog_G: [batch, time, heads] - Gradient with respect to log discount factors

    Shape Restrictions:
        - Time dimension must be a multiple of 4
        - Feature dimension must be a multiple of 8
    
    The 0th (batch) and 2nd (heads) dimensions are treated as batch dimensions.
    The discount factors are broadcasted along the final feature dimension.
    """
    dX, dlog_G = discumsum_bwd_cuda(log_G, dout, out)
    return dX, dlog_G

# Fake implementation for tracing and testing
@discumsum_bwd.register_fake
def discumsum_bwd_fake(dout, out, log_G):
    b, n1, h, d = dout.shape
    n = n1 - 1
    return (torch.empty(b, n, h, d, device=dout.device, dtype=dout.dtype),
            torch.empty(b, n, h, device=dout.device, dtype=torch.float32))

# Useful function to create sample inputs
def create_inputs(b=2, n=4, h=8, d=16, X_dtype=torch.float16, device='cuda'):
    n1 = n + 1
    generator = torch.Generator(device=device).manual_seed(42)
    dout = torch.randn(size=(b, n1, h, d), dtype=X_dtype, device=device, generator=generator) / d**.25
    out = torch.randn(size=(b, n1, h, d), dtype=X_dtype, device=device, generator=generator) / d**.25
    log_G = torch.zeros(size=(b, n, h), dtype=torch.float32, device=device) - .01
    return dout, out, log_G


## TUTORIAL ##
if __name__ == '__main__':
    # Hyperparameters
    b, n, h, d = (2, 4, 8, 16)
    dtype = torch.float16
    # Create inputs
    dout, out, log_G = create_inputs(b, n, h, d, dtype, 'cuda')
    # Run function
    with torch.no_grad():
        dX, dlog_G = discumsum_bwd(dout, out, log_G)
    # Compile function, fullgraph=True confirms no graph breaks
    compiled_discumsum_bwd = torch.compile(discumsum_bwd, fullgraph=True)
    with torch.no_grad():
        for _ in range(3):
            dX, dlog_G = compiled_discumsum_bwd(dout, out, log_G)
