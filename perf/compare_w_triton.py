import pandas as pd

from retention.create_inputs import create_inputs
from retention.triton import power_full as power_full_triton
from retention.vidrial import power_full as power_full_vidrial
from vidrial.kernels.sympow_mma.dimensions import sympow_dim
from vidrial.py_utils.common import default_d_tile
from vidrial.jit.settings import settings, PickBest
from perf._timing import estimate_runtime
import logging

logging.basicConfig(level=logging.DEBUG)

def flops(b, t, h, d, chunk_size, switch_over_seq_len, deg=2):
    if t <= switch_over_seq_len:
        return b * h * (t * t * d + t * d * d * 2)
    else:
        n, c = t // chunk_size, chunk_size
        D = sympow_dim(d, deg, d_tile=default_d_tile(d, deg))
        return b * h * n * (c * c * d + c * d * d * 2 + d * D * c * 2 + c * d * D * 2)


def main():
    results, chunk_size, switch_over_seq_len = [], 128, 512
    for b, t, h, d in [
        (1, 1024, 8, 32),
        (1, 1024, 8, 64),
        # (1, 1024, 8, 128),
        (8, 1024, 8, 32),
        (8, 1024, 8, 64),
        # (8, 1024, 8, 128),
        (1, 4096, 8, 32),
        (1, 4096, 8, 64),
        # (1, 4096, 8, 128),
        (8, 4096, 8, 32),
        (8, 4096, 8, 64),
        # (8, 4096, 8, 128),
        (1, 8192, 8, 32),
        (1, 8192, 8, 64),
        # (1, 8192, 8, 128),
        (8, 8192, 8, 32),
        (8, 8192, 8, 64),
        # (8, 8192, 8, 128),
    ]:
        inputs = create_inputs(b=b, t=t, h=h, d=d, switch_over_seq_len=switch_over_seq_len, chunk_size=chunk_size)
        triton_runtime = estimate_runtime(power_full_triton, num1=3, num2=10, **inputs)
        with settings.set(policy=PickBest, max_configs=999999, max_workers=127):
            vidrial_runtime = estimate_runtime(power_full_vidrial, num1=3, num2=10, **inputs)
        total_flops = flops(b, t, h, d, chunk_size, switch_over_seq_len)
        flops_triton = total_flops / triton_runtime * 1e-9
        flops_vidrial = total_flops / vidrial_runtime * 1e-9
        print(f'{b=}, {t=}, {h=}, {d=}, {triton_runtime=}, {vidrial_runtime=}, {flops_triton=}, {flops_vidrial=}')
        results.append({
            'b': b,
            't': t,
            'h': h,
            'd': d,
            'triton_runtime': triton_runtime,
            'vidrial_runtime': vidrial_runtime,
            'flops_triton': flops_triton,
            'flops_vidrial': flops_vidrial,
        })
    df = pd.DataFrame(results)
    print(df)
    df.to_csv('power_full_compare_w_triton.csv', index=False)



if __name__ == '__main__':
    main()