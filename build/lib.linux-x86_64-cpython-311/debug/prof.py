import math
import torch
from torch.utils._pytree import tree_map
from state_kernel.interface import SymmetricStateKernel
from flash_attn_manifest.flash_attn_interface import flash_attn_func, SYMPOWER, SOFTMAX
from einops import rearrange
from test_all import Config


def profile_power(cfg: Config):
    kernel = SymmetricStateKernel(cfg)
    b, t, h, d = cfg.shape
    Q = torch.randn((b, t, h, d), dtype=cfg.dtype, device='cuda')
    K = torch.randn((b, t, h, d), dtype=cfg.dtype, device='cuda')
    V = torch.randn((b, t, h, d), dtype=cfg.dtype, device='cuda')
    kernel(Q, K, V, attention_only=True)

def profile_sdpa(cfg: Config):
    b, t, h, d = cfg.shape
    Q = torch.randn((b, h, t, d), dtype=cfg.dtype, device='cuda')
    K = torch.randn((b, h, t, d), dtype=cfg.dtype, device='cuda')
    V = torch.randn((b, h, t, d), dtype=cfg.dtype, device='cuda')
    torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=True)
    
def profile_flash(cfg: Config):
    b, t, h, d = cfg.shape
    Q = torch.randn((b, t, h, d), dtype=cfg.dtype, device='cuda')
    K = torch.randn((b, t, h, d), dtype=cfg.dtype, device='cuda')
    V = torch.randn((b, t, h, d), dtype=cfg.dtype, device='cuda')
    flash_attn_func(Q, K, V, dropout_p=0.0, softmax_scale=1.0, causal=True, window_size=(0, 0), similarity=SOFTMAX, deg=4, softcap=0, alibi_slopes=None, deterministic=False, return_attn_probs=False)

def profile_chunk_state(cfg: Config):
    kernel = SymmetricStateKernel(cfg)
    Q, K, V = kernel.create_inputs(chunked=True)
    kernel.chunk_states(K, V)


def profile_query_state(cfg: Config):
    kernel = SymmetricStateKernel(cfg)
    Q, K, V = kernel.create_inputs(chunked=False)
    Q = rearrange(Q, 'b (n c) h d -> b n c h d', c=cfg.chunk_size)
    b, n, c, h, d = Q.shape
    D = math.comb(cfg.d + cfg.p - 1, cfg.p)
    S = torch.randn((b, n, h, D, d), dtype=Q.dtype, device=Q.device)
    norm = torch.randn((b, n, h, D), dtype=torch.float32, device=Q.device)
    kernel.query_states(Q, S, norm)


if __name__ == "__main__":
    cfg = Config([1, 65536, 1, 64], 1024, 2, torch.bfloat16, torch.float32, 1e-5)
    profile_chunk_state(cfg)
    # profile_query_state(cfg)
    # b, t, h, d = cfg.shape
    # D = math.comb(cfg.d + cfg.p - 1, cfg.p)
    # CHUNK_STATE_BLOCK_D = 128
    # CHUNK_STATE_BLOCK_T = 16

    # chunk_state_v_read_bytes = b * t / cfg.chunk_size * ((D + CHUNK_STATE_BLOCK_D - 1)// CHUNK_STATE_BLOCK_D) * ((cfg.chunk_size / CHUNK_STATE_BLOCK_T) + 1) * CHUNK_STATE_BLOCK_T * d * 2
    # chunk_state_v_read_requests = chunk_state_v_read_bytes / 32 / 16

    # QUERY_STATE_BLOCK_D = 16
    # QUERY_STATE_BLOCK_T = 128
    # query_state_d_read_bytes = b * t / cfg.chunk_size * (cfg.chunk_size // QUERY_STATE_BLOCK_T) * ((D + QUERY_STATE_BLOCK_D - 1) // QUERY_STATE_BLOCK_D) * QUERY_STATE_BLOCK_D * d * 2
    # query_state_d_read_requests = query_state_d_read_bytes / 32 / 16

    # print(f"chunk_state_v_read_bytes: {chunk_state_v_read_bytes}")
    # print(f"chunk_state_v_read_requests: {chunk_state_v_read_requests}")
    # print(f"query_state_d_read_bytes: {query_state_d_read_bytes}")
    # print(f"query_state_d_read_requests: {query_state_d_read_requests}")
    

    # profile_sdpa(cfg)
    # profile_flash(cfg)
    # profile_power(cfg)

