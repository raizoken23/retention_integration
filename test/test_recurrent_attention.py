import code
import math
import pytest
import torch
from einops import rearrange
from packages.power_attention.test.utils import *
from packages.power_attention.power_attention.power_full import PowerAttentionKernel
from torch.utils._pytree import tree_map

@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('seqlen_q', [16, 37, 64, 128, 134, 256, 257, 1024])
@pytest.mark.parametrize('num_heads', [1, 2, 3, 8, 12, 24])
@pytest.mark.parametrize('head_size', [32, 64])
@pytest.mark.parametrize('chunk_size', [None, 128, 256, 1024])
@pytest.mark.parametrize('p', [2])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('seed', [42])
@pytest.mark.parametrize('gating', [True, False])
@pytest.mark.parametrize('ε', [1e-5])
@pytest.mark.parametrize('critical_length', [None, 8192])
def test_recurrent_attention_fwd_and_bwd(batch_size, seqlen_q, num_heads, head_size, chunk_size, p, dtype, seed, gating, ε, critical_length):
    torch.manual_seed(seed)
    torch.set_printoptions(precision=7, sci_mode=False, edgeitems=1000, linewidth=10000)

    if chunk_size and chunk_size > seqlen_q:
        pytest.skip("Chunk size must be <= sequence length")
    if chunk_size and seqlen_q % chunk_size != 0:
        pytest.skip("Sequence must be divisible by chunk size")

    power_attention = PowerAttentionKernel(head_size, p, ε, dtype)
    power_attention_gold = PowerAttentionKernel(head_size, p, ε, torch.float32)
    inputs = create_QKVR(batch_size, seqlen_q, num_heads, head_size, dtype=dtype, gating=gating, log_gating=False)

    inputs_gold = paramify(pytree_to(inputs, torch.float32))
    O_gold = power_attention_gold(*inputs_gold, chunk_size=chunk_size, critical_length=critical_length, use_reference=True) # gold is quadratic attention fp32 reference
    loss_gold = O_gold.norm()
    loss_gold.backward()

    inputs_ref = paramify(inputs)
    O_ref = power_attention(*inputs_ref, chunk_size=chunk_size, critical_length=critical_length, use_reference=True)
    loss_ref = O_ref.norm()
    loss_ref.backward()

    inputs = paramify(inputs)
    O = power_attention(*inputs, chunk_size=chunk_size, critical_length=critical_length)
    loss = O.norm()
    loss.backward()

    print(f'{batch_size=}, {seqlen_q=}, {num_heads=}, {head_size=}, {chunk_size=}, {p=}, {dtype=}, {seed=}, {gating=}, {ε=}')
    compare(O_gold, O, O_ref, precision=1e-4, verbose_name='O')
    compare(inputs_gold[0].grad, inputs[0].grad, inputs_ref[0].grad, allowance=3.5, precision=1e-4, verbose_name='dQ')
    compare(inputs_gold[1].grad, inputs[1].grad, inputs_ref[1].grad, allowance=3.5, precision=1e-4, verbose_name='dK')
    compare(inputs_gold[2].grad, inputs[2].grad, inputs_ref[2].grad, allowance=3.5, precision=1e-4, verbose_name='dV')
    if gating:
        compare(inputs_gold[3].grad, inputs[3].grad, inputs_ref[3].grad, allowance=3.5, precision=1e-4, verbose_name='dR')
