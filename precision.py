import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from test.utils import *
from power_attention.power_full import PowerAttentionKernel

# from power_attention.power_attention import PowerAttentionKernel


def get_precision(ref_val, cuda_val, torch_val, precision=1e-3, verbose_name=None):
    cuda_ref_diff = torch.abs(ref_val - cuda_val)
    torch_ref_diff = torch.abs(ref_val - torch_val)
    cuda_precision = (cuda_ref_diff < precision).float().mean()
    torch_precision = (torch_ref_diff < precision).float().mean()
    if verbose_name:
        print(f'Cuda {verbose_name} precision: {cuda_precision}')
        print(f'Torch {verbose_name} precision: {torch_precision}')
    return cuda_precision.item(), torch_precision.item()


def fwd_precision(batch_size, seqlen_q, num_heads, head_size, chunk_size, p, dtype, seed, gating, ε, precision=1e-3):
    print(f'batch_size: {batch_size}, seqlen_q: {seqlen_q}, num_heads: {num_heads}, head_size: {head_size}, chunk_size: {chunk_size}, p: {p}, dtype: {dtype}, seed: {seed}, gating: {gating}, ε: {ε}')
    torch.manual_seed(seed)

    power_attention = PowerAttentionKernel(head_size, p, ε, dtype)
    power_attention_gold = PowerAttentionKernel(head_size, p, ε, torch.float32)
    inputs = create_QKVR(batch_size, seqlen_q, num_heads, head_size, dtype=dtype, gating=gating, log_gating=True)

    inputs_gold = paramify(pytree_to(inputs, torch.float32))
    O_gold = power_attention_gold(*inputs_gold, use_reference=True) # gold is quadratic attention fp32 reference

    inputs_ref = paramify(inputs)
    O_ref = power_attention(*inputs_ref, chunk_size=chunk_size, use_reference=True)

    inputs = paramify(inputs)
    O = power_attention(*inputs, chunk_size=chunk_size)

    return get_precision(O_gold, O, O_ref, precision=1e-3)


def run(cached=False):
    batch_size = 1
    num_heads = 1
    chunk_size = 2**10
    p = 2
    seed = 42
    ε = 1e-5
    precision = 1e-3
    GATING = [True, False]
    DTYPE = [torch.float16, torch.bfloat16]
    HEAD_SIZE = [32, 64]
    num_seqlen = 4
    SEQLEN = np.logspace(10, 10 + num_seqlen - 1, num=num_seqlen, base=2).astype(np.int32)

    data = []

    if not cached:
        for dtype in DTYPE:
            for gating in GATING:
                for head_size in HEAD_SIZE:
                    for seqlen in SEQLEN:
                        cuda_precision, torch_precision = fwd_precision(batch_size, seqlen, num_heads, head_size, chunk_size, p, dtype, seed, gating, ε, precision=precision)
                        data.append({
                            'seqlen': seqlen,
                            'head_size': head_size,
                            'chunk_size': chunk_size,
                            'p': p,
                            'dtype': 'fp16' if dtype == torch.float16 else 'bf16',
                            'gating': gating,
                            'cuda_precision': cuda_precision,
                            'torch_precision': torch_precision,
                            'label': f'{dtype} {gating} {head_size} {p}',
                        })
        df = pd.DataFrame(data)
        df.to_csv('precision.csv', index=True)
    else:
        df = pd.read_csv('precision.csv')

    # plot precision subplots
    num_dtypes = len(DTYPE)
    num_gating = len(GATING)
    num_head_sizes = len(HEAD_SIZE)
    fig, axs = plt.subplots(num_gating * num_head_sizes, num_dtypes, figsize=(12, 10), layout="constrained")
    for (i, ((gating, head_size, dtype), group)), ax in zip(enumerate(df.groupby(['gating', 'head_size', 'dtype'])), axs.flatten()):
        group.plot(x='seqlen', y='cuda_precision', ax=ax, label='Cuda Precision')
        group.plot(x='seqlen', y='torch_precision', ax=ax, ls='--', label='Torch Precision')
        ax.set_title(f'{gating=} {head_size=} {dtype=}')
        ax.set_yscale('linear')
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        
    fig.suptitle('Precision Comparison: Cuda vs Torch Implementation', fontsize=16)
    plt.savefig('precision.png')


if __name__ == '__main__':
    run(True)
