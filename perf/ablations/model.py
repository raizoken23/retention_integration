import torch
import torch.nn as nn
from power_attention.vidrial_fused import power_full, power_full_inference
from flash_attn.flash_attn_interface import flash_attn_func

class PowerAttention(nn.Module):

    def __init__(self, num_heads, hidden_size, chunk_size, degree, head_size, qhead_ratio, gating, device, dtype, kernel):
        super().__init__()
        self.n_head = num_heads
        self.n_embd = hidden_size
        self.chunk_size = chunk_size
        self.degree = degree
        self.head_size = head_size
        self.qhead_ratio = qhead_ratio
        self.gating = gating
        self.device = device
        self.dtype = dtype
        self.kernel = kernel

        # key, query, value projections for all heads, but in a batch
        self.q_dim = self.n_head * qhead_ratio * self.head_size
        self.k_dim = self.n_head * self.head_size
        self.v_dim = self.n_head * self.head_size
        self.gating_size = num_heads if self.gating else 0
        # self.c_attn = nn.Linear(hidden_size, self.qkv_size + self.gating_size, bias=True)
        self.q_proj = nn.Linear(hidden_size, self.q_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.k_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.v_dim, bias=False)
        self.gating_proj = nn.Linear(hidden_size, self.gating_size, bias=False)
        # output projection
        self.c_proj = nn.Linear(qhead_ratio * num_heads * head_size, hidden_size, bias=True)

    def forward(self, hidden_states, past_key_values=None, use_cache=False):
        B, T, C = hidden_states.size() # batch size, sequence length, embedding dimensionality (n_embd)
        h = self.n_head
        hq = self.qhead_ratio * h
        d = self.head_size

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.q_proj(hidden_states)
        g = self.gating_proj(hidden_states)
        if not use_cache:
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            k = k.view(B, T, h, d)
            v = v.view(B, T, h, d)
            state = None
        else:
            k = past_key_values[0]['k'] # [B, cache_len, h, d]
            v = past_key_values[0]['v'] # [B, cache_len, h, d]
            state = past_key_values[0]['state'] # [B, h, D, d]
            g = past_key_values[0]['g'] if self.kernel == 'power' else None # [B, cache_len, h]

        q = q.view(B, T, hq, d)

        if self.gating and self.kernel == 'power':
            # 6.906768 corresponds to initial gating of .999
            log_g = torch.nn.functional.logsigmoid(6.906768 + g.to(dtype=torch.float32)).contiguous()
        else:
            log_g = None

        if use_cache and self.kernel == 'power':
            y, state = power_full_inference(q.contiguous(), k.contiguous(), v.contiguous(), log_g, state,
                deg=self.degree,
                scale=1.0 / d**0.5,
                chunk_size=self.chunk_size)                
        elif self.kernel == 'power':
            y = power_full(q.contiguous(), k.contiguous(), v.contiguous(), log_g,
                deg=self.degree,
                scale=1.0 / d**0.5,
                chunk_size=self.chunk_size)
        else:
            y = flash_attn_func(q.contiguous(), k.contiguous(), v.contiguous(), causal=True)

        y = y.contiguous().view(B, T, hq * d) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        if use_cache and self.kernel == 'power':
            return y, state
        else:
            return y