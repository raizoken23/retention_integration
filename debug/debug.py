import os, socket
import torch
import torch.distributed as dist
from einops import rearrange
from state_kernel.interface import KernelConfig, SymmetricStateKernel
from torch.utils._pytree import tree_map

HOSTNAME = socket.gethostname()
LOCAL_SIZE = int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
LOCAL_RANK = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_SIZE'])
WORLD_RANK = int(os.environ['OMPI_COMM_WORLD_RANK'])
DEVICE = torch.device("cuda:{}".format(LOCAL_RANK))

class SequenceParallelSymmetricStateKernel(SymmetricStateKernel):
    pass

if __name__ == '__main__':
    dist.init_process_group('nccl',
                            rank=WORLD_RANK,
                            world_size=WORLD_SIZE)

    torch.cuda.device(DEVICE)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(LOCAL_RANK)

    print(f'{HOSTNAME = }, '
          f'{LOCAL_SIZE = }, '
          f'{LOCAL_RANK = }, '
          f'{WORLD_SIZE = }, '
          f'{WORLD_RANK = }, '
          f'{dist.get_rank() = } '
          f'{dist.get_world_size() = }')

    b = 1
    h = 1
    t = 4096
    d = 32
    dtype = torch.float16

    Q = torch.randn(b, t, h, d, dtype=dtype).cuda(DEVICE)
    K = torch.randn(b, t, h, d, dtype=dtype).cuda(DEVICE)
    V = torch.randn(b, t, h, d, dtype=dtype).cuda(DEVICE)

    spssk = SymmetricStateKernel(KernelConfig(
        chunk_size=512,
        d=d,
        p=2,
        ε=1e-6,
        dtype=dtype,
        acc_dtype=torch.float32
    )).cuda(DEVICE)

    with torch.inference_mode():
        print(f'{LOCAL_RANK = } {Q.device = } {K.device = } {V.device = }')
        Y = spssk(Q,K,V)
        print(f'success on {Y.device}')