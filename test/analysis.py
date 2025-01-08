import math

class Config:
    def __init__(self, d, p, blockD, blockT, T):
        self.d = d
        self.p = p
        self.blockD = blockD
        self.blockT = blockT
        self.T = T

def mix_parallel(cfg):
    T = cfg.T
    d = cfg.d
    p = cfg.p
    D = math.comb(d + p - 1, p)
    blockD = cfg.blockD
    blockT = cfg.blockT
    nwarps = blockD * blockT // (16 * 8)
    copy_size = 8
    res_size = 8
    groups_per_block = (blockD + res_size - 1) // res_size
    ngroups = (D + res_size - 1) // res_size
    groups_per_warp = 32 // blockT

    total_smem_copy_requests = 0
    total_smem_copy_elements = 0
    midx = [0] * p
    cached_dim = set()
    for g in range(ngroups):
        counter = 0
        for _ in range(res_size):
            if midx[-1] not in cached_dim:
                total_smem_copy_requests += 1
                total_smem_copy_elements += copy_size
                counter += 1
                cache_dim_start = (midx[-1] // copy_size) * copy_size
                cached_dim = set(range(cache_dim_start, cache_dim_start + copy_size))
            j = p - 1
            while j >= 0 and midx[j] == d - 1:
                j -= 1
            if j < 0:
                print(f"OOB encountered at group {g}")
                break
            else:
                midx[j] += 1
                for k in range(j + 1, p):
                    midx[k] = midx[k - 1]
        print(f"group {g}: {midx}, {counter} reads")
        cached_dim = set()

    print(f"blockD: {blockD}, blockT: {blockT}, nwarps: {nwarps}")
    print(f"groups_per_block: {groups_per_block}, ngroups: {ngroups}, groups_per_warp: {groups_per_warp}")
    print(f"Total SMEM copy requests: {total_smem_copy_requests * T / copy_size}")
    print(f"Total SMEM copy elements: {total_smem_copy_elements * T}")


if __name__ == "__main__":
    cfg = Config(64, 2, 128, 8, 65536)
    mix_parallel(cfg)

