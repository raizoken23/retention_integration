# Power Attention


# Development

Power Attention uses c++17 (because torch only supports at most c++17).

## Build

Power Attention uses setuptools as the build backend. It uses Torch's CUDA extension to compile the CUDA code. 

To build the wheel (which consists of building the extension, packaging the python bindings, and creating the wheel), a Python environment where **torch** is installed is required. 

If you are using **poetry**, you can activate the environment and build the wheel with the following commands:
```
poetry shell
python setup.py bdist_wheel
```

or, equivalently,
```
poetry shell
make build
```

## Install

To install the wheel, run
```
pip install dist/state_kernel-<version>-cp311-cp311-linux_x86_64.whl
```

or, equivalently,
```
make install
```
This will install the package to the active Python environment. 



# Math

## Attention Form

```
Q: [t, d]
K: [t, d]
V: [t, d]
O: [t, d]

S = QK^T
M = Causal Mask
T = p * log(abs(S) + ε)
Z = T + p * (G_Q @ 1^T - 1 @ G_K^T)
P = exp(Z * M)
Y = P @ V
y = P @ 1
O = Y / y[:, None]
```

Complexity

```
S = QK^T: O(t^2 * d)
T = p * log(abs(S) + ε): O(t^2)
Z = T + p * (G_Q @ 1^T - 1 @ G_K^T): O(t^2)
P = exp(Z * M): O(t^2)
Y = P @ V: O(t^2 * d)
y = P @ 1: O(t^2)
O = Y / y[:, None]: O(t * d)
```

Total: 

```
O(2t^2d + 4t^2 + td)
```


## Recurrent Form

```
Q: [t/c, c, d]
K: [t/c, c, d]
V: [t/c, c, d]
O: [t, d]
c: chunk size

Phi_K = embed(K) # [t/c x c x D], O(t/c * c * D)
S = Phi_K.transpose(1, 2) @ V^T  # [t/c x D x d], O(t/c * D * d * c)
N = Phi_K.transpose(1, 2) @ 1  # [t/c x D], O(t/c * D * c)
Phi_Q = embed(Q) # [t/c x c x D], O(t/c * c * D)
Y_chunk = Phi_Q @ S  # [t/c x c x d], O(t/c * c * d * D)
y_chunk = Phi_Q @ N  # [t/c x c], O(t/c * c * D)

S = QK^T: O(t/c * (c)^2 * d)
T = p * log(abs(S) + ε): O(t/c * (c)^2)
Z = T + p * (G_Q @ 1^T - 1 @ G_K^T): O(t/c * (c)^2)
P = exp(Z * M): O(t/c * (c)^2)
Y_attn = P @ V: O(t/c * (c)^2 * d)
y_attn = P @ 1: O(t/c * (c)^2)

O = (Y_chunk + Y_attn) / (y_chunk + y_attn)[:, None]  # [t/c x c x d], O(t/c * c * d)
```

Total: 
```
O(tD + tDd + tD + tD + tDd + tD + tcd + tc + tc + tc + tcd + tc + td)
= O(2tDd + 4tD + 2tcd + 4tc + td)
```

## Complexity difference

```
(2t^2d + 4t^2 + td) - (2tDd + 4tD + 2tcd + 4tc + td)
= 2t^2d + 4t^2 + td - 2tDd - 4tD - 2tcd - 4tc - td
= 2t^2d + 4t^2 - 2tDd - 4tD - 2tcd - 4tc
= 2t(td - 2t - Dd - 2D - cd - 2c)
= 2t(t(d - 2) - D(d - 2) - c(d - 2))
= 2t(d - 2)(t - D - c)
```

This means, when `t > D + c`, the attention form should be slower. Conversely, the attention form should be faster when `t < D + c`. Here we can define `D + c` as the cross-over point, a context size at which an efficient linear attention should switch from attention form to recurrent form. 

We want this cross-over point to be as small as possible to benefit from the reduced complexity of the linear form. We can see from the expression that the smaller `c` is, the smaller the cross-over point is. However, if `c` is too small, the number of chunks `t/c` increases, leading to higher memory consumption. 


# Benchmark

