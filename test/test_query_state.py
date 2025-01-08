import math
import pytest
import torch
from einops import rearrange
from packages.state_kernel.test.utils import *
from packages.state_kernel.state_kernel.query_state import symmetric_power_query_state, symmetric_power_query_state_reference, InnerBlock, OuterBlock
from torch.utils._pytree import tree_map

@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('seqlen_q', [128, 256, 1024])
@pytest.mark.parametrize('num_heads', [1, 2, 5, 8, 12])
@pytest.mark.parametrize('head_size', [32, 64])
@pytest.mark.parametrize('chunk_size', [128, 256, 1024])
@pytest.mark.parametrize('p', [2])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('gating', [False])
@pytest.mark.parametrize('seed', [44])
@pytest.mark.parametrize('ε', [1e-5])
def test_query_state_fwd_and_bwd(batch_size, seqlen_q, num_heads, head_size, chunk_size, p, dtype, seed, gating, ε):
    torch.manual_seed(seed)
    # torch.set_printoptions(precision=7, sci_mode=False, edgeitems=1000, linewidth=10000)
    _4GB = 4096 * 1024 * 1024

    NumOuterBlocks = head_size // OuterBlock
    NumInnerBlocks = head_size // InnerBlock
    D = ((InnerBlock // OuterBlock + NumOuterBlocks) * NumInnerBlocks // 2) * (InnerBlock * OuterBlock)
    S_size = D * head_size

    if chunk_size > seqlen_q:
        pytest.skip("Chunk size must be <= sequence length")

    if seqlen_q % chunk_size != 0:
        pytest.skip("Sequence length must be divisible by chunk size")

    num_chunks = seqlen_q // chunk_size
    if num_chunks * num_heads * batch_size * S_size * 2 * 16 > _4GB:  # 10 accounts for ref + torch + cuda, forward + backward
        pytest.skip(f"Memory limit exceeded {_4GB / 1024 / 1024 / 1024:.2f} GB, skipping")

    if p == 4 and num_chunks * num_heads * batch_size > 1:
        pytest.skip("Skipping test for p=4 and large sequence lengths, will be tested in full tests")

    Q,_,_,_ = create_QKVR(batch_size, seqlen_q, num_heads, head_size,
                         chunk_size=chunk_size, dtype=dtype, gating=gating, log_gating=True)
    S,s = create_SN(batch_size, seqlen_q // chunk_size, num_heads, head_size, D, dtype)
    inputs = Q, S, s

    inputs_gold = paramify(pytree_to(inputs, torch.float32))
    Y_gold, y_gold = symmetric_power_query_state_reference(*inputs_gold, p, float(D) if dtype == torch.float16 else None)
    loss_gold = (Y_gold / (y_gold[..., None] + ε)).norm()
    loss_gold.backward()

    inputs_ref = paramify(inputs)
    Y_ref, y_ref = symmetric_power_query_state_reference(*inputs_ref, p, float(D) if dtype == torch.float16 else None)
    loss_ref = (Y_ref / (y_ref[..., None] + ε)).norm()
    loss_ref.backward()

    inputs = paramify(inputs)
    Y_buffer = torch.empty((batch_size, num_chunks, chunk_size, num_heads, head_size), dtype=dtype, device=Q.device)
    y_buffer = torch.empty((batch_size, num_chunks, chunk_size, num_heads), dtype=torch.float32, device=Q.device)
    log_G = torch.randn((batch_size, num_chunks, chunk_size, num_heads), dtype=torch.float32, device=Q.device) - 1
    Q.requires_grad = True
    S.requires_grad = True
    s.requires_grad = True
    Y, y = symmetric_power_query_state(Q, S, s, None, None, log_G, p, float(D) if dtype == torch.float16 else None, ε)
    loss = (Y / (y[..., None])).norm()
    loss.backward()

    # compare(Y_gold, Y, Y_ref, precision=1e-3, verbose_name='Y')
    # compare(y_gold, y, y_ref, abs_tol=1e-2, precision=1e-3, verbose_name='y')
    # compare(inputs_gold[0].grad, inputs[0].grad, inputs_ref[0].grad, precision=1e-5, verbose_name='dQ')
    # compare(inputs_gold[1].grad, inputs[1].grad, inputs_ref[1].grad, precision=1e-5, verbose_name='dS')
    # compare(inputs_gold[2].grad, inputs[2].grad, inputs_ref[2].grad, precision=1e-5, verbose_name='ds')

