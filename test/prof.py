import torch
import os

from einops import rearrange
from packages.state_kernel.state_kernel.power_full import PowerAttentionKernel
from state_kernel.chunk_state import symmetric_power_chunk_state, ExpandedDim
from state_kernel.query_state import symmetric_power_query_state
from state_kernel.attention import symmetric_power_attention
from torch.utils._pytree import tree_map
import torch.nn.functional as F

from flash_attn import flash_attn_func
from state_kernel_cuda import discumsum, discumsum_bwd

def create_QKVR(b, t, h, d, dtype, gating=False, log_gating=True, chunk_size=None, device='cuda'):
    """Create random Q, K, V tensors, optionally with gating coefficients"""
    Q = (
        torch.rand((b, t, h, d), dtype=dtype)
        .to(device)
    )
    K = (
        torch.rand((b, t, h, d), dtype=dtype)
        .to(device)
    )
    V = (
        torch.rand((b, t, h, d), dtype=dtype)
        .to(device)
    )
    if chunk_size is not None:
        Q, K, V = tree_map(
            lambda X: rearrange(
                X, 'b (n c) h d -> b n c h d', c=chunk_size
            ),
            (Q, K, V),
        )
    if gating:
        R = torch.rand([b, t, h], dtype=torch.float32).to(device) * 10
        if log_gating:
            log_Γ = F.logsigmoid(R.detach()).cumsum(1)
            if chunk_size is not None:
                log_G = rearrange(log_Γ, 'b (n c) h -> b n c h', c=chunk_size)
            else:
                log_G = log_Γ
            R = log_G
        elif chunk_size is not None:
            R = rearrange(R, 'b (n c) h -> b n c h', c=chunk_size)
    else:
        R = None
    return Q, K, V, R

# compare the perf between power and flash
def compare_attention():
    head_size = 64
    p = 2
    ε = 1e-5
    dtype = torch.float16
    gating = True
    chunk_size = 2048
    critical_length = None
    token_count = 131072
    seqlen_q = 2048
    batch_size = token_count // seqlen_q
    num_heads = 4

    torch.manual_seed(42)

    # create 1 chunk per batch
    inputs = create_QKVR(batch_size, seqlen_q, num_heads, head_size, dtype, gating, chunk_size=seqlen_q, log_gating=False)

    Q, K, V, R = inputs
    Q.requires_grad_(True)
    K.requires_grad_(True)
    V.requires_grad_(True)
    if gating:
        R.requires_grad_(True)

    log_G = F.logsigmoid(R).cumsum(1)

    # power attention
    Y_attn, y_attn = symmetric_power_attention(Q, K, V, log_G, p, 1.0, ε)
    (Y_attn.norm() + y_attn.norm()).backward(retain_graph=True)

    # flash attention
    Q, K, V = tree_map(lambda x: rearrange(x, 'b n c h d -> b (n c) h d'), inputs[:3])
    Y_flash = flash_attn_func(Q, K, V, dropout_p=0.0, causal=True)
    Y_flash.norm().backward(retain_graph=True)


def query_state_ablate_state_expansion(Q, S, s, Y_attn, y_attn, log_G, p, ε):
    head_size = 64
    p = 2
    ε = 1e-5
    dtype = torch.float16
    gating = False
    chunk_size = 2048
    critical_length = None
    token_count = 65536
    batch_size = 4
    num_heads = 1
    seqlen_q = token_count // batch_size # 32768
    num_chunks = seqlen_q // chunk_size # 32

    torch.manual_seed(42)

    inputs = create_QKVR(batch_size, seqlen_q, num_heads, head_size, dtype, gating, chunk_size=chunk_size, log_gating=False)

    Q, K, V, R = inputs
    Q.requires_grad_(True)
    K.requires_grad_(True)
    V.requires_grad_(True)
    if gating:
        R.requires_grad_(True)

    D = ExpandedDim(head_size, p)

    # query_state with state expansion
    S = torch.randn((batch_size, num_chunks, num_heads, D, head_size), dtype=dtype).to(Q.device)
    s = torch.randn((batch_size, num_chunks, num_heads, D), dtype=torch.float32).to(Q.device)
    Y_attn = torch.randn((batch_size, num_chunks, chunk_size, num_heads, head_size), dtype=dtype).to(Q.device)
    y_attn = torch.randn((batch_size, num_chunks, chunk_size, num_heads), dtype=torch.float32).to(Q.device)
    log_G = torch.randn((batch_size, num_chunks, chunk_size, num_heads), dtype=torch.float32).to(Q.device)

    Y_qs, y_qs = symmetric_power_query_state(Q, S, s, Y_attn, y_attn, log_G, p, None, ε)
    (Y_qs.norm() + y_qs.norm()).backward(retain_graph=True)


def chunk_state_ablation():
    head_size = 64
    p = 2
    ε = 1e-5
    dtype = torch.float16
    gating = False
    chunk_size = 2048
    critical_length = None
    token_count = 131072
    batch_size = token_count // chunk_size
    num_heads = 4
    seqlen_q = chunk_size # 2048
    num_chunks = seqlen_q // chunk_size # 16

    torch.manual_seed(42)

    inputs = create_QKVR(batch_size, seqlen_q, num_heads, head_size, dtype, gating, chunk_size=chunk_size, log_gating=False)

    Q, K, V, R = inputs
    Q.requires_grad_(True)
    K.requires_grad_(True)
    V.requires_grad_(True)
    if gating:
        R.requires_grad_(True)

    D = ExpandedDim(head_size, p)


    Y_qs, y_qs = symmetric_power_chunk_state(K, V, 2)
    (Y_qs.norm() + y_qs.norm()).backward(retain_graph=True)





def run_discumsum():

    batch_size = 1024
    num_chunks = 16
    num_heads = 1
    D = 1024
    dtype = torch.float16

    X = torch.randn((batch_size, num_chunks, num_heads, D), dtype=dtype, device='cuda')
    Y = torch.zeros((batch_size, num_chunks + 1, num_heads, D), dtype=dtype, device='cuda')
    discount = torch.randn(batch_size, num_chunks, num_heads, dtype=torch.float32, device='cuda') - 1

    out = discumsum(X, discount, Y)

    out_grad = torch.randn_like(out)
    dX, dD = discumsum_bwd(discount, out_grad, out)



if __name__ == '__main__':

    chunk_state_ablation()
    compare_attention()


# import torch
# from packages.state_kernel.state_kernel.power_attention import PowerAttentionKernel

# def f(Q, K, V, log_G, Y_grad, d):
#     pa = PowerAttentionKernel(d, 2, 0.001, torch.float16)
#     Y = pa(Q, K, V, log_G, chunk_count=n)
#     # return Y
#     Y.backward(Y_grad)


# b, n, c, h, D, d = (16, 4, 1024, 32, 2560, 64)
# t = n*c
# Q, K, V = (torch.empty(size=(b, t, h, d), dtype=torch.float16, device='cuda', requires_grad=True)  for _ in range(3))
# log_G = torch.randn(size=(b, t, h), dtype=torch.float32, device='cuda', requires_grad=True) - 1
# Y_grad = torch.empty(size=(b, t, h, d), dtype=torch.float16, device='cuda')
# f_ = torch.compile(f)
# for _ in range(2):
#     print(" ------ ", _)
#     f_(Q, K, V, log_G, Y_grad, d)
# print("Done")