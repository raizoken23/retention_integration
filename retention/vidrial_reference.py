import torch
from retention.power_retention_maker import make_power_retention_fused, make_power_retention_inference
from retention._attention.reference import attention
from retention._update_state.reference_vidrial_fused import update_state
from retention._discumsum.reference import discumsum_reference as discumsum
from retention._query_state.reference_vidrial_fused import query_state

power_retention = make_power_retention_fused(update_state, query_state, discumsum, attention)
power_retention_inference = make_power_retention_inference(update_state, query_state, attention)


## TUTORIAL ##
if __name__ == '__main__':
    from perf._inspect import print_runtime
    from retention.create_inputs import create_inputs

    # Create inputs
    t = 1024
    chunk_size=128
    b = 2
    h = 2
    d = 64
    deg = 2
    gating = True
    dtype = torch.bfloat16
    inputs = create_inputs(b=b, t=t, h=h, d=d, dtype=dtype, device='cuda', gating=gating, chunk_size=chunk_size, deg=deg, requires_grad=True)
    
    import sys

    O = power_retention(**inputs)
    torch.autograd.backward((O,), grad_tensors=(O,))

    if len(sys.argv) > 1 and sys.argv[1] == 'benchmark':
        # Benchmark
        print(f"Benchmarking power_retention {b=} {t=} {h=} {d=} {chunk_size=} {deg=} {gating=} {dtype=}")

        print_runtime(power_retention, **inputs)