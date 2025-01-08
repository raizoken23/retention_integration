# Build

```
cd experiment
mkdir build
cd build
cmake ..
make
```

# Run

```
./state_expand/state_expand <T> <p> <d>
```
where `<T>`, `<p>`, and `<d>` are the chunk size, order of power, and feature size, respectively.

For example,
```
manifest2-py3.11sean@jbox:~/manifest2/packages/state_kernel/csrc/experiment/build$ ./state_expand/state_expand 131072
T = 131072
p = 4
d = 32
D = 52360
Using device 0: NVIDIA RTX A6000  (SM86, 84 SMs)
CUTE_GEMM:     [29395.8]GFlop/s  (14.9418)ms
```


# What does this code do?

It does the following:
1. Create a K, V, and S matrix, put random values in them.
2. Time a kernel that's supposed to compute the state expansion, where the kernel does the following:
    - Load a BlockTxHeaddim Chunk of K into shared memory
    - Compute the state expansion for that chunk in register (which currently does nothing)
    - Load a BlockTxHeaddim Chunk of V into shared memory
    - Do a matmul between the expanded state (in register) and the V chunk (in shared memory)
    - Store the result S in shared memory, and then global memory

# Why is this code here?

This code is here to hopefully allow Carles and Jacob to have a easier time making changes to the state expansion algorithm.



