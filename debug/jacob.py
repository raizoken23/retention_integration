import code
import gc

import torch
import torch.nn
from torch.autograd import Function

from wip.minitorch.profiling.gpu_mem_track import MemTracker
from wip.minitorch.profiling.utils import tsize_mb
from torch.utils.checkpoint import checkpoint as rematerialize_on_backward

class AttentionTenpow2(torch.nn.Module):
    p = 2
    acc_dtype = torch.float32
    ε = 1e-5
    bQ = 64
    bK = 64
    bKV = 64
    bQS = 64

    def forward(self, Q, K, V, chunk_count):
        def to_chunked(X):
            b, t, h, d = X.shape
            assert t % chunk_count == 0, f'{t=} not divisible by {chunk_count=}'
            return X.view([b, chunk_count, t // chunk_count, h, d]).permute(1,0,3,2,4)
        chunked_Q = to_chunked(Q)
        chunked_K = to_chunked(K)
        chunked_V = to_chunked(V)

        local_S, local_s = self.chunk_keyvalue_to_state(chunked_K, chunked_V)
        S, s = self.merge_states(local_S, local_s)

        local_Y,local_Z = self.chunk_query_keyvalue(chunked_Q, chunked_K, chunked_V)
        state_Y,state_Z = self.chunk_query_state(chunked_Q[1:], S[:-1], s[:-1])

        chunked_Y = local_Y; chunked_Y[1:] += state_Y
        chunked_Z = local_Z; chunked_Z[1:] += state_Z

        out = (chunked_Y / chunked_Z[...,None]).permute(1,0,3,2,4).reshape(V.shape)

        return out

    def query_keyvalue_growing_memory(self, Q, K, V):
        # compute preattention
        D = Q @ K.T
        # change dtype
        D = D.type(self.acc_dtype)
        # power
        C = D ** self.p
        # mask
        B = torch.where(torch.tril(torch.ones(C.shape, dtype=torch.bool, device=C.device)), C, 0)
        # unnormalized attend
        Y = B.type(V.dtype) @ V
        # normalizer
        Z = (B.sum(-1, dtype=self.acc_dtype) + self.ε).type(V.dtype)
        return Y, Z

    def chunk_query_keyvalue(self, Q, K, V):
        chunks, batch_size, head_count, seq_len, _ = Q.shape
        bQ, bK = self.bQ, self.bK

        Y = torch.zeros(chunks, batch_size, head_count, seq_len, V.shape[-1], dtype=V.dtype, device=V.device)
        Z = torch.zeros(chunks, batch_size, head_count, seq_len, dtype=V.dtype, device=V.device) + self.ε

        for q_start in range(0, seq_len, bQ):
            q_end = min(q_start + bQ, seq_len)
            Q_block = Q[:, :, :, q_start:q_end, :]
            for k_start in range(0, q_end, bK):
                k_end = min(k_start + bK, q_end)

                K_block = K[:,:,:,k_start:k_end,:]
                V_block = V[:,:,:,k_start:k_end,:]

                # Compute preattention for a block of queries and keys
                D_block = Q_block @ K_block.transpose(-2, -1)
                D_block = D_block.type(self.acc_dtype)

                # Apply power
                C_block = D_block ** self.p

                # mask
                mask = (torch.arange(q_start, q_end, device=C_block.device)[None,None,None,:,None] >=
                        torch.arange(k_start, k_end, device=C_block.device)[None,None,None,None,:])
                B_block = torch.where(mask, C_block, 0)

                # Accumulate unnormalized attention and normalizer
                Y[:,:,:,q_start:q_end, :] += B_block.type(V.dtype) @ V_block
                Z[:,:,:,q_start:q_end] += B_block.sum(-1, dtype=self.acc_dtype)

        return Y, Z

    def keyvalue_to_state_growing_memory(self, K_block, V_block):
        s = K_block.transpose(3,4) @ K_block
        expanded_K_block = K_block[...,None,:] * K_block[...,:,None]
        S = torch.einsum('abcdef, abcdg -> abcefg', expanded_K_block, V_block)

        return S, s

    def Φ(self, Q_or_K):
        return torch.outer(Q_or_K, Q_or_K)

    def chunk_keyvalue_to_state(self, K, V):
        chunks, batch_size, head_count, seq_len, K_feature_size = K.shape
        chunks, batch_size, head_count, seq_len, V_feature_size = V.shape

        # Initialize the KK and KKV accumulators
        s = torch.zeros(chunks, batch_size, head_count, K_feature_size, K_feature_size,
                         dtype=K.dtype, device=K.device)
        S = torch.zeros(chunks, batch_size, head_count, K_feature_size, K_feature_size, V_feature_size,
                          dtype=V.dtype, device=V.device)

        # Iterate over the sequence dimension in blocks
        for start in range(0, seq_len, self.bKV):
            end = min(start + self.bKV, seq_len)

            K_block = K[:, :, :, start:end, :]
            V_block = V[:, :, :, start:end, :]

            inner_S, inner_s = self.keyvalue_to_state_growing_memory(K_block, V_block)
            s.add_(inner_s)
            S.add_(inner_S)

        return S, s

    def query_state_growing_memory(self, Q_block, S, s):
        QQ = Q_block[...,None,:] * Q_block[...,:,None]
        numerator = torch.einsum('abcijk, abcjkl -> abcil', QQ, S)
        denominator = torch.einsum('abcijk, abcjk -> abci', QQ, s)
        return numerator, denominator

    def chunk_query_state(self, Q, S, s):
        chunks, batch_size, head_count, seq_len, Q_feature_size = Q.shape
        V_feature_size = S.shape[-1]

        # Initialize accumulators for numerator and denominator
        numerator_accum = torch.zeros(chunks, batch_size, head_count, seq_len, V_feature_size,
                                      dtype=Q.dtype, device=Q.device)
        denominator_accum = torch.zeros(chunks, batch_size, head_count, seq_len,
                                        dtype=Q.dtype, device=Q.device)

        # Iterate over the sequence dimension in blocks
        for start in range(0, seq_len, self.bQS):
            end = min(start + self.bQS, seq_len)
            Q_block = Q[:,:,:,start:end, :]

            # Call the inner function for the current block
            numerator_block, denominator_block = self.query_state_growing_memory(Q_block, S, s)

            # Accumulate the results
            numerator_accum[:,:,:, start:end] = numerator_block
            denominator_accum[:,:,:, start:end] = denominator_block

        return numerator_accum, denominator_accum

    def merge_states(self, stacked_S, stacked_s):
        return (stacked_S.cumsum(0, dtype=self.acc_dtype).type_as(stacked_S),
                stacked_s.cumsum(0, dtype=self.acc_dtype).type_as(stacked_s))

if __name__ == '__main__':
    b = 1
    h = 1
    t = 32
    d = 16

    Q = torch.randn(b, t, h, d)
    K = torch.randn(b, t, h, d)
    V = torch.randn(b, t, h, d)

    def to_chunked(X, chunk_count):
        b, t, h, d = X.shape
        assert t % chunk_count == 0
        return X.reshape([b, chunk_count, t//chunk_count, h, d]).permute(1, 0, 3, 2, 4)
    Q_chunked = to_chunked(Q, 1)
    K_chunked = to_chunked(K, 1)
    V_chunked = to_chunked(V, 1)

    att = AttentionTenpow2()

    Y, Z = att.chunk_query_keyvalue(Q_chunked, K_chunked, V_chunked)
    gold = Y/Z[...,None]

    for chunk_count in [32, 16, 8, 4, 2, 1]:
        Y = att.forward(Q, K, V, chunk_count)
        print((torch.abs(Y[0,:,0] - gold[0,0,0]) < 1e-3).float().mean())