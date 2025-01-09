# Even split




# Uneven split

Given a problem size (the size of $K$) of $[b, n, c, h, d]$, we are going to launch a grid of $[b, n * h, d]$ blocks, where each block does the following.

Suppose head_dim = 64, deg/power = 4, we have a set of multi-indices $\mathcal{M}$ where
$$
|M| = \binom{64 + 4 - 1}{4}
$$

The set of multi-indices will be split out in the following way

$$
---\text{block }0---\\
[0, 0, 0, 0]\\
[0, 0, 0, 1]\\
...\\
[0, 0, 0, 63]\\
[0, 0, 1, 1]\\
...\\
[0, 0, 1, 63]\\
[0, 0, 2, 2]\\
...\\
[0, d, d, d]\\
---\text{block }1---\\
[1, 1, 1, 1]\\
[1, 1, 1, 2]\\
..\\
[1, 1, 1, d]\\
...\\
[1, d, d, d]\\
---\text{block }2---\\
[2, 2, 2, 2]\\
...\\
[2, d, d, d]\\
---\text{block }3---\\
...\\
---\text{block }63---\\
$$

Notice that each block/CTA will have different amount of workloads. This is fine because there's no block-level synchronization anyway.

Inside each block, we'll determine the number of warps using heuristics, but basically we just split the work among each warp using the second index.

$$
---\text{block }k---\\
--\text{warp }0--\\
[k, k, k, k]\\
[k, k, k, k+1]\\
...\\
[k, k, k, d]\\
[k, k, k+1, k+1]\\
...
[k, k, d, d]\\
--\text{warp }1--\\
[k, k+1, k+1, k+1]\\
[k, k+1, k+1, k+2]\\
...\\
[k, k+1, k+1, d]\\
[k, k+1, k+2, k+2]\\
...
[k, k+1, d, d]\\
--\text{warp }2--\\
[k, k+2, k+2, k+2]\\
...\\
--\text{warp }3--\\
[k, k+3, k+3, k+3]\\
...\\
--\text{warp }0--\\
[k, k+4, k+4, k+4]\\
...
$$

If there're $N$ warps in block $k$, warp $j$ is responsible for $\sum_{g=0}^{(d-k)//N}[k, k+j+g*N, *, *]$. Simply put, warp $j$ is reponsible for a group of indices where the second left-most index is in $\{x>= k \mid (x - k) \mod N \equiv j\}$

Note that for blocks with $k > d - 4$, it's possible that some warps will have no work at all. This is also fine because they are just not going to get scheduled, nothing is wasted.

Inside each warp, we are going to split the work based on the right-most index, which will ineivitably create some wasted cycles, but much better than if we split the work based on left-most index.

Suppose we are at block $k$, warp $j$, we have the following layout

$$
\begin{aligned}
---\text{block }&k---\\
--\text{warp }j&, \text{group } g--\\
-\text{thread }&i-\\
[k, k + gN + j, &k + gN + j, k + gN + j + i]\\
[k, k + gN + j, &k + gN + j, k + gN + j + i + 32]\\
[k, k + gN + j, &k + gN + j + 1, k + gN + j + 1 + i]\\
[k, k + gN + j, &k + gN + j + 1, k + gN + j + 1 + i + 32]\\
..&.\\
[k, k + gN + j, &d, k + gN + j + i]\\
[k, k + gN + j, &d, k + gN + j + i + 32]\\
\end{aligned}
$$

Each thread will be responsible for
$$
\sum_{h=0}^{d-k-gN-j} (d - k - gN - j - i) // 32
$$

As a result of this layout, it's ineivitable that at later part of the inner loop, many threads will have no data to compute. In the extreme case, in the last iteration, the will only be one thread in the whole warp.

However, I made this design decision because this allows minimal memory access. Due to ALU's theoretical 1 cycle per output performance (after latency hiding, see [here](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions)), compared to share memory's 20+ cycles per read, 