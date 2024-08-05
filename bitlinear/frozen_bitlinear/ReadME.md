## Setup
Make sure you have the correct packages installed in your virtual environment. In a conda environment, you can run:
``` 
> conda create -n env python pip
> conda activate env
> pip install -r requirements.txt
```

Afterwards, you need to build the desired CUDA kernels for use on your machine. To do so, you can selectively build the kernels you would like to test or use through:
``` 
> cd path/to/bitlinear/bitlinear/frozen_bitlinear/cuda/pack_weights
> python setup.py build_ext --inplace
> cd path/to/bitlinear/bitlinear/frozen_bitlinear/cuda/kernels/*
> python setup.py build_ext --inplace
```
You can also build all of them at once by running 
``` 
> cd path/to/bitlinear/bitlinear/frozen_bitlinear/
> chmod +x scripts/build.sh
> scripts/build.sh
```

## Testing
Choose one of the kernels available in ```path/to/bitlinear/bitlinear/frozen_bitlinear/src/``` to test against the PyTorch baseline for your device. 
``` 
< conda activate env
< cd path/to/bitlinear/bitlinear/frozen_bitlinear
< chmod +x scripts/test.sh
< scripts/test.sh 
    -d <device> (CUDA_AVAILABLE_DEVICES=$device)
    -k <kernel_name> 
```

All logs, data, and plots are stored locally under ```path/to/bitlinear/bitlinear/frozen_bitlinear/results/{%Y%m%d_%T}/```.

If any issues come up, you can reach me at ```sopsahl@mit.edu```.

## Cleanup 
To clean the builds and results, run the following commands.
``` 
> cd path/to/bitlinear/bitlinear/frozen_bitlinear
> chmod +x scripts/clean.sh
> scripts/clean.sh 
    -b (default: builds only)
    -w (weights)
    -r (results)
```









## Motivations

Matrix multiplications are a key building block of most modern high-performance computing systems.
They are notoriously hard to optimize, hence their implementation is generally done by
hardware vendors themselves as part of so-called "kernel libraries" (e.g., cuBLAS).
Unfortunately, these libraries are often proprietary and cannot be easily customized
to accommodate the needs of modern deep learning workloads (e.g., fused activation functions).
In comes Triton, which is easily customizeable and works to fit our need.

Roughly speaking, the traditional kernel implements the following blocked
algorithm to multiply a (M, K) by a (K, N) matrix:

```python
# Do in parallel
for m in range(0, M, BLOCK_SIZE_M):
  # Do in parallel
  for n in range(0, N, BLOCK_SIZE_N):
    acc = zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
    for k in range(0, K, BLOCK_SIZE_K):
      a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
      b = B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]
      acc += dot(a, b)
    C[m : m+BLOCK_SIZE_M, n : n+BLOCK_SIZE_N] = acc
```
where each iteration of the doubly-nested for-loop is performed by a dedicated Triton program instance.

In a linear instance, all that is changed is the addition of the bias on the final step.

## Compute Kernel

The above algorithm is, actually, fairly straightforward to implement in Triton.
The main difficulty comes from the computation of the memory locations at which blocks
of ```A``` and ```B``` must be read in the inner loop. For that, we need
multi-dimensional pointer arithmetic.

### Pointer Arithmetic

For a row-major 2D tensor `X`, the memory location of `X[i, j]` is given
by `&X[i, j] = X + i*stride_xi + j*stride_xj`.
Therefore, blocks of pointers for `A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K]` and
`B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]` can be defined in pseudo-code as:

```python
&A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K] =  a_ptr + (m : m+BLOCK_SIZE_M)[:, None]*A.stride(0) + (k : k+BLOCK_SIZE_K)[None, :]*A.stride(1)
&B[k : k+BLOCK_SIZE_K, n:n+BLOCK_SIZE_N] =  b_ptr + (k : k+BLOCK_SIZE_K)[:, None]*B.stride(0) + (n : n+BLOCK_SIZE_N)[None, :]*B.stride(1)
```

### L2 Cache Optimizations

As mentioned above, each program instance computes a `[BLOCK_SIZE_M, BLOCK_SIZE_N]`
block of `C`.
It is important to remember that the order in which these blocks are computed does
matter, since it affects the L2 cache hit rate of our program, and unfortunately, a
simple row-major ordering

```Python
pid = tl.program_id(axis=0)
grid_n = tl.cdiv(N, BLOCK_SIZE_N)
pid_m = pid // grid_n
pid_n = pid % grid_n
```

is just not going to cut it in the traditional case. When we are evaluating the bitlinear case, however, we must remember that the much more expensive operation is the activation load, not the weight load. We need to find balance between a simple row-major ordering that only requires one load for each block on `axis=0`, and one that optimizes over grouped clocking methods.