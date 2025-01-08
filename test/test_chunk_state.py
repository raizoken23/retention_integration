import math
import pytest
import torch
from einops import rearrange
from packages.state_kernel.test.utils import *
from packages.state_kernel.state_kernel.chunk_state import symmetric_power_chunk_state, symmetric_power_chunk_state_reference, InnerBlock_DT as InnerBlock, OuterBlock_DT as OuterBlock
from torch.utils._pytree import tree_map

@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('seqlen_q', [128, 256, 1024])
@pytest.mark.parametrize('num_heads', [1, 2, 3, 4])
@pytest.mark.parametrize('head_size', [32, 64])
@pytest.mark.parametrize('chunk_size', [128, 256, 1024])
@pytest.mark.parametrize('p', [2])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('seed', [44])
@pytest.mark.parametrize('gating', [False])
@pytest.mark.parametrize('ε', [1e-5])
@pytest.mark.parametrize('memory_limit', [4096 * 1024 * 1024]) # 4GB
def test_chunk_state_fwd_and_bwd(batch_size, seqlen_q, num_heads, head_size, chunk_size, p, dtype, seed, gating, ε, memory_limit):
    torch.manual_seed(seed)
    # torch.set_printoptions(precision=7, sci_mode=False, edgeitems=1000, linewidth=10000)

    NumOuterBlocks = head_size // OuterBlock
    NumInnerBlocks = head_size // InnerBlock
    D = ((InnerBlock // OuterBlock + NumOuterBlocks) * NumInnerBlocks // 2) * (InnerBlock * OuterBlock)
    S_size = D * head_size
    if (seqlen_q / chunk_size) * num_heads * batch_size * S_size * 2 * 16 > memory_limit:  # 10 accounts for ref + torch + cuda, forward + backward
        pytest.skip(f"Memory limit exceeded {memory_limit / 1024 / 1024 / 1024:.2f} GB, skipping")

    if p == 4 and (seqlen_q / chunk_size) * num_heads * batch_size > 1:
        pytest.skip("Skipping test for p=4 and large sequence lengths, will be tested in full tests")

    if chunk_size > seqlen_q:
        pytest.skip("Chunk size must be <= sequence length")

    _,K,V,_ = create_QKVR(batch_size, seqlen_q, num_heads, head_size,
                         chunk_size=chunk_size, dtype=dtype, gating=gating, log_gating=True)
    inputs = K,V

    inputs_gold = paramify(pytree_to(inputs, torch.float32))
    S_gold, s_gold = symmetric_power_chunk_state_reference(*inputs_gold, p)
    loss_gold = S_gold.mean() / s_gold.mean()
    loss_gold.backward()

    inputs_ref = paramify(inputs)
    S_ref, s_ref = symmetric_power_chunk_state_reference(*inputs_ref, p)
    loss_ref = S_ref.mean() / s_ref.mean()
    loss_ref.backward()

    inputs = paramify(inputs)
    S, s = symmetric_power_chunk_state(*inputs, p)
    loss = S.mean() / s.mean()
    loss.backward()

    compare(S_gold, S, S_ref, precision=1e-3, verbose_name='S')
    compare(s_gold, s, s_ref, abs_tol=1e-2, precision=1e-3, verbose_name='s')
    compare(inputs_gold[1].grad, inputs[1].grad, inputs_ref[1].grad, precision=1e-5, verbose_name='dV')
    compare(inputs_gold[0].grad, inputs[0].grad, inputs_ref[0].grad, precision=1e-5, verbose_name='dK')
