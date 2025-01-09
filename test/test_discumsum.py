import torch
import pytest
from test.utils import *
from power_attention_cuda import discumsum, discumsum_bwd


def ref_discumsum(X, discount, out):
    out_shape = out.shape
    B, N_1, H, D = out_shape
    N = N_1 - 1

    assert X.shape == (B, N, H, D)
    assert out.shape == (B, N_1, H, D)
    assert discount.shape == (B, N, H)

    acc = out[:, 0, ...]
    res = [acc.clone()]
    for n in range(0, N):
        acc = acc * torch.exp(discount[:, n, ...]).unsqueeze(-1).to(X.dtype) + X[:, n, ...]
        res.append(acc.clone())
    
    real_out = torch.cat(res, dim=1)
    return real_out.view(out_shape)


def ref_discumsum_bwd(discount, dout, out):
    # discount: (B, N, H)
    # dout: (B, N+1, H, D)
    # out: (B, N+1, H, D)
    # dX: (B, N, H, D)
    dout_shape = dout.shape
    B, N_1, H, D = dout_shape
    N = N_1 - 1

    assert dout.shape == out.shape
    assert discount.shape == (B, N, H)
    assert out.shape == (B, N+1, H, D)

    dX = torch.empty((B, N, H, D), dtype=dout.dtype, device=dout.device) # (B, N, H, D)
    dD = torch.empty((B, N, H), dtype=discount.dtype, device=dout.device)

    dX[:, -1] = dout[:, -1]
    dD[:, -1] = (out[:, -2] * dX[:, -1]).sum(-1) * torch.exp(discount[:, -1])
    for n in range(N - 2, -1, -1):
        dX[:, n] = dout[:, n + 1] + dX[:, n+1] * torch.exp(discount[:, n+1, :]).unsqueeze(-1)
        dD[:, n] = ((out[:, n] * dX[:, n]).sum(-1)) * torch.exp(discount[:, n])
    
    return dX, dD


@pytest.mark.parametrize('batch_size', [1])
@pytest.mark.parametrize('num_chunks', [4, 16, 64, 68, 128, 256])
@pytest.mark.parametrize('num_heads', [1, 2])
@pytest.mark.parametrize('D', [16, 64, 128, 200, 256, 1024, 2048, 2560])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_discumsum_fwd(batch_size, num_chunks, num_heads, D, dtype):
    torch.manual_seed(42)

    X = torch.randn((batch_size, num_chunks, num_heads, D), dtype=dtype, device='cuda')
    Y = torch.zeros((batch_size, num_chunks + 1, num_heads, D), dtype=dtype, device='cuda')
    discount = torch.randn(batch_size, num_chunks, num_heads, dtype=torch.float32, device='cuda') - 1

    gold_X = X.to(torch.float32)
    gold_Y = Y.to(torch.float32)
    ref_X = X.clone()
    ref_Y = Y.clone()

    ref_out = ref_discumsum(ref_X, discount, ref_Y)
    gold_out = ref_discumsum(gold_X, discount, gold_Y)
    out = discumsum(X, discount, Y)

    # to_csv(gold_out, 'gold_out.csv')
    # to_csv(out, 'out.csv')
    # to_csv(ref_out, 'ref_out.csv')
    
    compare(gold_out, out, ref_out, abs_tol=1e-3, precision=1e-3, verbose_name='out')


    
@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('num_chunks', [4, 16, 32, 64, 68, 128])
@pytest.mark.parametrize('num_heads', [1, 2])
@pytest.mark.parametrize('D', [16, 64, 128, 200, 256, 1024, 2048, 2560])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_discumsum_bwd(batch_size, num_chunks, num_heads, D, dtype):
    torch.manual_seed(42)
    X = torch.randn((batch_size, num_chunks, num_heads, D), dtype=dtype, device='cuda', requires_grad=True)
    Y = torch.ones((batch_size, num_chunks + 1, num_heads, D), dtype=dtype, device='cuda', requires_grad=True)

    discount = torch.randn(batch_size, num_chunks, num_heads, dtype=torch.float32, device='cuda', requires_grad=True) - 2
    X.retain_grad()
    Y.retain_grad()
    discount.retain_grad()

    copy = lambda x, dtype: x.clone().to(dtype).requires_grad_(True)
    X_gold, Y_gold, discount_gold = copy(X, torch.float32), copy(Y, torch.float32), copy(discount, torch.float32)
    X_gold.retain_grad()
    Y_gold.retain_grad()
    discount_gold.retain_grad()
    gold_out = ref_discumsum(X_gold, discount_gold, Y_gold)
    gold_out.retain_grad()
    gold_out.norm().backward()
    gold_dX = X_gold.grad
    gold_dD = discount_gold.grad
    
    X_ref, discount_ref, Y_ref = copy(X, dtype), copy(discount, torch.float32), copy(Y, dtype)
    X_ref.retain_grad()
    discount_ref.retain_grad()
    Y_ref.retain_grad()
    ref_out = ref_discumsum(X_ref, discount_ref, Y_ref)
    ref_out.retain_grad()
    ref_out.norm().backward()
    ref_dX = X_ref.grad
    ref_dD = discount_ref.grad

    ref_manual_dX, ref_manual_dD = ref_discumsum_bwd(discount_ref.detach().clone(), ref_out.grad.detach().clone(), ref_out.detach().clone())

    cuda_dX, cuda_dD = discumsum_bwd(discount_gold.detach().clone(), gold_out.grad.detach().clone().to(dtype), gold_out.detach().clone().to(dtype))

    # to_csv(discount_gold[0, :, 0], 'discount_gold.csv')
    # to_csv(gold_out[0, :, 0], 'gold_out.csv')
    # to_csv(gold_out.grad[0, :, 0], 'gold_out_grad.csv')
    # to_csv(gold_dX[0, :, 0], 'gold_dX.csv')
    # to_csv(cuda_dX[0, :, 0], 'cuda_dX.csv')
    # to_csv(ref_dX[0, :, 0], 'ref_dX.csv')
    # to_csv(gold_dD[0, :, 0], 'gold_dD.csv')
    # to_csv(cuda_dD[0, :, 0], 'cuda_dD.csv')
    # to_csv(ref_dD[0, :, 0], 'ref_dD.csv')
    # to_csv(ref_manual_dX[0, :, 0], 'ref_manual_dX.csv')
    # to_csv(ref_manual_dD[0, :, 0], 'ref_manual_dD.csv')

    compare(gold_dX, ref_manual_dX, ref_dX, abs_tol=1e-5, precision=1e-3, verbose_name='dX')
    compare(gold_dD, ref_manual_dD, ref_dD, abs_tol=1e-5, precision=1e-3, verbose_name='dD')
    compare(gold_dX, cuda_dX, ref_dX, abs_tol=1e-5, precision=1e-3, verbose_name='dX')
    compare(gold_dD, cuda_dD, ref_dD, abs_tol=1e-5, precision=1e-3, verbose_name='dD')
    