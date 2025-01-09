import torch

def create_test_tensors(batch_size=4, seqlen=16384, chunk_size=2048, nheads=32, headdim=64, device='cuda', dtype=torch.bfloat16):
    """Create test tensors for power attention benchmarking."""
    # Create q, k, v tensors
    qkv = torch.randn(batch_size, seqlen//chunk_size, chunk_size, 3, nheads, headdim, device=device, dtype=dtype, requires_grad=True)
    q, k, v = qkv.unbind(dim=3)
    log_g = torch.nn.functional.logsigmoid(torch.randn(batch_size, seqlen//chunk_size, chunk_size, nheads, device=device, dtype=torch.float32, requires_grad=True))
    return q, k, v, log_g

def get_compiled_versions(fn, *args, warmup=3, **kwargs):
    """Takes a function and args andreturns compiled versions for fwd, bwd, and fwd+bwd passes.
    
    Args:
        fn: Function to compile
        
    Returns:
        Tuple of (fwd_fn, bwd_fn, fwd_bwd_fn)
    """
    # Run forward pass to identify output shapes
    args = tuple(arg.detach().requires_grad_() if isinstance(arg, torch.Tensor) else arg for arg in args)
    out = fn(*args, **kwargs)
    # Create random grad tensors matching output shapes
    if isinstance(out, tuple):
        grad_tensors = tuple(torch.randn_like(x) for x in out)
    else:
        grad_tensors = torch.randn_like(out)
    # Define functions
    def fwd():
        with torch.no_grad():
            return fn(*args, **kwargs)
    torch._dynamo.config.compiled_autograd = True
    def bwd():        
        torch.autograd.backward(out, grad_tensors=grad_tensors, retain_graph=True)
    torch._dynamo.config.compiled_autograd = False
    def fwd_bwd():
        out = fn(*args, **kwargs)
        torch.autograd.backward(out, grad_tensors=grad_tensors)
    # Compile functions
    compiled_fwd = torch.compile(fwd, dynamic=False)
    compiled_bwd = torch.compile(bwd, dynamic=False)
    compiled_fwd_bwd = torch.compile(fwd_bwd, dynamic=False)
    # Warmup passes
    for _ in range(warmup):
        compiled_fwd()
        compiled_bwd()
        compiled_fwd_bwd()
    # Return compiled functions
    return (compiled_fwd, compiled_bwd, compiled_fwd_bwd)

def wrap_with_timer(fn, n=10):
    """Takes a function and returns a function that calls it n times and returns the total time."""
    def timed_fn(*args, **kwargs):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(n):
            out = fn(*args, **kwargs)
        end_event.record()
        end_event.synchronize()
        return out, start_event.elapsed_time(end_event)
    return timed_fn

def estimate_runtime(fn, *args, num1=10, num2=30, **kwargs):
    """Takes a function and returns a an estimate of time per iteration."""
    timed_fn_1 = wrap_with_timer(fn, num1)
    timed_fn_2 = wrap_with_timer(fn, num2)

    _, t1 = timed_fn_1(*args, **kwargs)
    _, t2 = timed_fn_2(*args, **kwargs)
        
    return (t2 - t1) / (num2 - num1)

def report_fwd_bwd(fn, *args, **kwargs):
    fwd_fn, bwd_fn, fwdbwd_fn = get_compiled_versions(fn, *args, **kwargs)
    fwd_time = estimate_runtime(fwd_fn)
    bwd_time = estimate_runtime(bwd_fn)
    fwdbwd_time = estimate_runtime(fwdbwd_fn)
    print(f"fwd_time: {fwd_time:.2f}ms, bwd_time: {bwd_time:.2f}ms, fwdbwd_time: {fwdbwd_time:.2f}ms")
