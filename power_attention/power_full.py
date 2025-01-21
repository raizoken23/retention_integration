## POWER FULL KERNEL ##
# Implements the power self-attention algorithm using CUDA kernels.

## IMPLEMENTATION ##
import torch
import torch.nn.functional as F
from enum import Enum
from functools import partial
from torch.utils._pytree import tree_map
from power_attention._attention import attention, attention_reference
from power_attention._update_state import update_state, update_state_reference
from power_attention._discumsum import discumsum, discumsum_reference
from power_attention._query_state import query_state, query_state_reference
from power_attention._p1_full import p1_full, _update_state as update_state_matmul, _query_state as query_state_matmul
from power_attention._config import normal_space
from power_attention._utils import compute_expanded_dim
import math


class UpdateStateImpl(Enum):
    CUTLASS = 0
    REFERENCE = 1
    MATMUL = 2

class QueryStateImpl(Enum):
    CUTLASS = 0
    REFERENCE = 1
    MATMUL = 2

class DiscumsumImpl(Enum):
    CUTLASS = 0
    REFERENCE = 1

class AttentionImpl(Enum):
    CUTLASS = 0
    REFERENCE = 1
    TRITON = 2

IMPL_MAP = {
    UpdateStateImpl.CUTLASS: update_state,
    UpdateStateImpl.REFERENCE: update_state_reference,
    UpdateStateImpl.MATMUL: update_state_matmul,
    QueryStateImpl.CUTLASS: query_state,
    QueryStateImpl.REFERENCE: query_state_reference,
    QueryStateImpl.MATMUL: query_state_matmul,
    DiscumsumImpl.CUTLASS: discumsum,
    DiscumsumImpl.REFERENCE: discumsum_reference,
    AttentionImpl.CUTLASS: attention,
    AttentionImpl.REFERENCE: attention_reference,
    # AttentionImpl.TRITON: attention_triton,
}

class PowerFullFactory:
    @staticmethod
    def pick_impl(impl: Enum):
        if impl not in IMPL_MAP:
            raise ValueError(f"Invalid implementation: {impl}")
        return IMPL_MAP[impl]

    @staticmethod
    def make_power_full(update_state_impl: UpdateStateImpl, query_state_impl: QueryStateImpl, discumsum_impl: DiscumsumImpl, attention_impl: AttentionImpl):

        def _power_full(Q, K, V, log_G=None, initial_state=None,
                    deg=2, scale=None, chunk_size=None): # noqa: C901
            """Compute power attention with optional gating and chunking.
            
            Implements the power attention algorithm with optional gating. On short sequences,
            this is equivalent to the quadratic power attention algorithm. On long sequences, we split 
            the sequence into chunks of size chunk_size (or chunk_n chunks) and process recurrently.

            Args:
                Q, K, V: torch.Tensor of shape [batch, seq, head, dim] - Query, Key and Value tensors
                log_G: Optional[torch.Tensor] of shape [batch, seq, head] - Optional log gating factors
                initial_state: Optional[Tuple[torch.Tensor]] - Not implemented, must be None
                deg: int - Power attention degree
                scale: Optional[float] - Optional multiplicative scale factor on attention scores (since recurrent form is equivalent to attention, this applies to recurrent form as well)
                chunk_size: Optional[int] - Size of each chunk

            Returns:
                torch.Tensor of shape [batch, seq, head, dim] - Output tensor

            Input restrictions:
                - Q, K, V must have same shape and dtype (32 or 64, and fp16 or bf16)
                - log_G must be float32 if provided
                - Must have at least 4 chunks
                - Sequence length must be evenly divisible by chunk count/size
            """
            if initial_state is not None:
                raise NotImplementedError('Initial state not implemented')

            if deg == 1: # when deg == 1, update_state and query_state are essentially matmuls
                _update_state = PowerFullFactory.pick_impl(UpdateStateImpl.MATMUL)
                _query_state = PowerFullFactory.pick_impl(QueryStateImpl.MATMUL)
            else:
                _update_state = PowerFullFactory.pick_impl(update_state_impl)
                _query_state = PowerFullFactory.pick_impl(query_state_impl)
            _discumsum = PowerFullFactory.pick_impl(discumsum_impl)
            _attention = PowerFullFactory.pick_impl(attention_impl)
            
            # Establish shapes and dtypes
            assert Q.dtype == K.dtype == V.dtype, 'dtypes of inputs must match'
            dtype = Q.dtype
            b, t, hq, d = Q.shape
            _, _, h, _ = K.shape
            assert hq % h == 0, f"Q heads must be a multiple of KV heads: {hq=} {h=}"
            qhead_ratio = hq // h
            if chunk_size is not None:
                c = chunk_size
                assert t % chunk_size == 0, f'{t=} not evenly divisible by {chunk_size=}'
                n = t // chunk_size
            else:
                c = t
                n = 1
            gating = log_G is not None
            if gating:
                log_G = log_G.to(torch.float32)

            if not scale:
                scale = 1.0

            # Quick return for simple quadratic attention
            if t <= c:
                if qhead_ratio > 1:
                    K = K.repeat_interleave(qhead_ratio, dim=2)
                    V = V.repeat_interleave(qhead_ratio, dim=2)
                    if gating:
                        log_G = log_G.repeat_interleave(qhead_ratio, dim=2)
                log_G_accum = log_G.cumsum(1) if log_G is not None else None
                Y, _, _ = _attention(Q, K, V, log_G_accum, deg, scale)
                assert Y.is_contiguous(), 'Y must be contiguous'
                return Y

            # Reshape into chunks
            Q = Q.view(b, n, c, hq, d)
            K = K.view(b, n, c, h, d)
            V = V.view(b, n, c, h, d)    
            if gating:
                log_G = log_G.view(b, n, c, h).to(torch.float32)
                log_G_intrachunk_accum = log_G.cumsum(2)

            # Compute chunk states
            if gating:
                log_discount_weights = (log_G_intrachunk_accum.narrow(2, c-1, 1) - log_G_intrachunk_accum) / deg
                cs_K = K * torch.exp(log_discount_weights).unsqueeze(-1).to(K.dtype)
            else:
                cs_K = K
            S = _update_state(cs_K.contiguous(), V.contiguous(), deg)

            # Accumulate
            if gating:
                log_G_chunk_sum = log_G_intrachunk_accum[:,:,-1].contiguous()
            else:
                log_G_chunk_sum = torch.zeros(size=(b, n, h), device=Q.device, dtype=torch.float32)
            S = _discumsum(S, log_G_chunk_sum) # Note that this adds an empty chunk to the start of the sequence
            S = S.narrow(1, 0, n)

            # Restrict to non-initial chunks for recurrent outputs
            Q = Q.contiguous()
            K = K.contiguous()
            V = V.contiguous()
            if gating:
                log_G_intrachunk_accum = log_G_intrachunk_accum.contiguous()

            # Compute attention
            Q_flatbatch = Q.view(b*n, c, hq, d)
            K_flatbatch = K.view(b*n, c, h, d)
            V_flatbatch = V.view(b*n, c, h, d)    
            log_G_intrachunk_accum_flatbatch = log_G_intrachunk_accum.view(b*n, c, h) if gating else None
            if qhead_ratio > 1:
                K_flatbatch = K_flatbatch.repeat_interleave(qhead_ratio, dim=2)
                V_flatbatch = V_flatbatch.repeat_interleave(qhead_ratio, dim=2)
                if gating:
                    log_G_intrachunk_accum_flatbatch = log_G_intrachunk_accum_flatbatch.repeat_interleave(qhead_ratio, dim=2)
            attn_Y, _, rowmax = _attention(Q_flatbatch, K_flatbatch, V_flatbatch, log_G_intrachunk_accum_flatbatch, deg, scale)
            attn_Y = attn_Y.view(b, n, c, hq, d)
            rowmax = rowmax.view(b, n, c, hq).detach()
            rowmax = rowmax - math.log(scale)

            if gating:
                if qhead_ratio > 1:
                    log_G_intrachunk_accum = log_G_intrachunk_accum.repeat_interleave(qhead_ratio, dim=3)
                Q = Q * torch.exp(log_G_intrachunk_accum / deg).unsqueeze(-1).to(Q.dtype)
            if qhead_ratio > 1:
                S = S.repeat_interleave(qhead_ratio, dim=2)
            D = float(compute_expanded_dim(d, deg))
            Y = _query_state(Q.contiguous(), S.contiguous(), attn_Y.contiguous(), rowmax.contiguous(), deg, D, initial_state is None)

            # Epilogue
            out = Y.contiguous().view(b, t, hq, d).to(dtype)
            return out

        return _power_full

power_full_reference = PowerFullFactory.make_power_full(UpdateStateImpl.REFERENCE, QueryStateImpl.REFERENCE, DiscumsumImpl.REFERENCE, AttentionImpl.REFERENCE)
power_full = PowerFullFactory.make_power_full(UpdateStateImpl.CUTLASS, QueryStateImpl.CUTLASS, DiscumsumImpl.CUTLASS, AttentionImpl.CUTLASS)

## Useful function to create sample inputs
def create_inputs(b=2, t=1024, h=8, d=32, qhead_ratio=1, dtype=torch.float16, device='cuda', gating=False,
                  chunk_size=None, deg=2, requires_grad=False, seed=42):
    torch.manual_seed(seed)
    Q = torch.randn(size=(b, t, h * qhead_ratio, d), dtype=dtype, device=device) / math.sqrt(d)
    K = torch.randn(size=(b, t, h, d), dtype=dtype, device=device) / math.sqrt(d)
    V = torch.randn(size=(b, t, h, d), dtype=dtype, device=device) / math.sqrt(d)
    if gating:
        log_G = F.logsigmoid(torch.randn(size=(b, t, h), dtype=torch.float32, device=device))
    else:
        log_G = None
    initial_state = None
    scale = (1./d)**deg
    if requires_grad:
        Q, K, V, log_G, initial_state = tree_map(
            lambda x: x.requires_grad_(True) if x is not None else None, (Q, K, V, log_G, initial_state))
    return dict(Q=Q, K=K, V=V, log_G=log_G, 
                initial_state=initial_state,
                deg=deg, scale=scale,
                chunk_size=chunk_size)


## TUTORIAL ##
if __name__ == '__main__':
    from perf._timing import report_fwd_bwd

    # Create inputs
    t = 1024
    chunk_size=128
    b = 8
    h = 16
    d = 64
    deg = 2
    gating = True
    dtype = torch.float16
    inputs = create_inputs(b=b, t=t, h=h, d=d, dtype=dtype, device='cuda', gating=gating, chunk_size=chunk_size, deg=deg, requires_grad=True)
    
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'profile':
        O = power_full(**inputs)
        torch.autograd.backward((O,), grad_tensors=(O,))
    else:
        # Benchmark
        print(f"Benchmarking power_full {b=} {t=} {h=} {d=} {chunk_size=} {deg=} {gating=} {dtype=}")

        report_fwd_bwd(power_full, **inputs)

