import math
import pytest
import torch
from einops import rearrange
from packages.power_attention.test.utils import *
from packages.power_attention.power_attention.attention import symmetric_power_attention, symmetric_power_attention_reference
from torch.utils._pytree import tree_map

@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('seqlen_q', [16, 37, 64, 128, 134, 256, 257, 1024])
@pytest.mark.parametrize('num_heads', [1, 2, 3, 8, 12, 24])
@pytest.mark.parametrize('head_size', [32, 64])
@pytest.mark.parametrize('p', [2, 4])
@pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.parametrize('seed', [42])
@pytest.mark.parametrize('gating', [True, False])
@pytest.mark.parametrize('ε', [1e-5])
def test_quadratic_attention(batch_size, seqlen_q, num_heads, head_size, p, dtype, seed, gating, ε):
    torch.manual_seed(seed)
    torch.set_printoptions(precision=7, sci_mode=False, edgeitems=1000, linewidth=10000)

    inputs = create_QKVR(batch_size, seqlen_q, num_heads, head_size,
                         chunk_size=seqlen_q, dtype=dtype, gating=gating, log_gating=True)

    inputs_gold = paramify(pytree_to(inputs, torch.float32))
    Y_gold, y_gold = symmetric_power_attention_reference(*inputs_gold, p, None, ε)
    O_gold = Y_gold / (y_gold[..., None] + ε)
    loss_gold = O_gold.norm()
    loss_gold.backward()

    inputs_ref = paramify(inputs)
    Y_ref, y_ref = symmetric_power_attention_reference(*inputs_ref, p, None, ε)
    O_ref = Y_ref / (y_ref[..., None] + ε)
    loss_ref = O_ref.norm()
    loss_ref.backward()

    inputs = paramify(inputs)
    Y, y = symmetric_power_attention(*inputs, p, None, ε)
    O = Y / (y[..., None] + ε)
    loss = O.norm()
    loss.backward()

    compare(O_gold, O, O_ref, precision=1e-3, verbose_name='O')
    compare(Y_gold, Y, Y_ref, precision=1e-3, verbose_name='Y')
    compare(y_gold, y, y_ref, precision=1e-3, verbose_name='y')
    compare(inputs_gold[0].grad, inputs[0].grad, inputs_ref[0].grad, precision=1e-3, verbose_name='dQ')
    to_csv(inputs[1].grad[0, 0, :, 0, :], 'dK.csv')
    compare(inputs_gold[1].grad, inputs[1].grad, inputs_ref[1].grad, precision=1e-3, verbose_name='dK')
    to_csv(inputs[2].grad[0, 0, :, 0, :], 'dV.csv')
    compare(inputs_gold[2].grad, inputs[2].grad, inputs_ref[2].grad, precision=1e-3, verbose_name='dV')
    if gating:
        compare(inputs_gold[3].grad, inputs[3].grad, inputs_ref[3].grad, precision=1e-3, verbose_name='dR')
