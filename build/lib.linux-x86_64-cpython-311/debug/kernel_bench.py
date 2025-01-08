# Contains utility to benchmarking kernel properly
# Reference: https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch
import os
import subprocess

import torch

DEVICE = os.environ.get("CUDA_VISIBLE_DEVICES")
CLOCK_SPEED = 1350  # Must choose a clock speed that's supported on your device.


def set_clock_speed(speed=CLOCK_SPEED):
    """
    Set GPU clock speed to a specific value.
    This doesn't guarantee a fixed value due to throttling, but can help reduce variance.
    """
    process = subprocess.Popen("nvidia-smi", stdout=subprocess.PIPE, shell=True)
    stdout, _ = process.communicate()
    process = subprocess.run(f'sudo nvidia-smi -pm ENABLED -i {DEVICE}', shell=True)
    process = subprocess.run(f"sudo nvidia-smi -lgc {CLOCK_SPEED} -i {DEVICE}", shell=True)


def reset_clock_speed():
    """
    Reset GPU clock speed to default values.
    """
    subprocess.run(f"sudo nvidia-smi -pm ENABLED -i {DEVICE}", shell=True)
    subprocess.run(f"sudo nvidia-smi -rgc -i {DEVICE}", shell=True)


# allocating 50MB to match L2 cache size on H100
x = torch.empty(int(50 * (1024**2)), dtype=torch.int8, device='cuda')


def flush_cache():
    x.zero_()


def time_kernel(fn, *inputs, steps=20, clock_speed=None, **kwinputs):
    """Time only the kernel run without launch overhead."""
    if clock_speed is not None:
        set_clock_speed(clock_speed)
    warmup_steps = steps // 10

    # Warmup steps
    for _ in range(warmup_steps):
        fn(*inputs, **kwinputs)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]

    for i in range(steps):
        # flush_cache()
        torch.cuda._sleep(1_000_000)

        start_events[i].record()
        output = fn(*inputs, **kwinputs)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    if clock_speed is not None:
        reset_clock_speed()
    return sum(times) / len(times), output
