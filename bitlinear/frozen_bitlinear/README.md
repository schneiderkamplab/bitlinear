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

If any issues come up, you can reach me at [sopsahl@mit.edu](mailto:sopsahl@mit.edu).

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
to accommodate the needs of our ternary system. Attempts at [matmul-free linear implementations](https://github.com/ridgerchu/matmulfreellm) still rely on the cuBLAS kernel for the matmul. The goal of this investigation is to develop a CUDA kernel competitive with cuBLAS in speed and performance, and ultimately far faster, as a ternary system can be implemented by a series of masked adds as opposed to traditional matrix multiplication.  
Roughly speaking, the traditional kernel implements the following blocked
algorithm to multiply a (M, K) by a (K, N) matrix:

```C++
  const uint col = blockIdx.x * blockDim.x + threadIdx.x;
  const uint row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < M && col < N) {

     for (int k = 0; k < K) {
        sum += A[row, k] * B[k, col];
      }

    output[row * N + col] = sum + bias[row];
  }
```
where each instance of the above code is a single thread.

We hope to develop a kernel that can avoid the multiplicative step, instead following the successive framework:

```C++
  const uint col = blockIdx.x * blockDim.x + threadIdx.x;
  const uint row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < M && col < N) {

     for (int k = 0; k < K) {
        if (B[k, col] == 1) {
            sum += A[row, k];
        } else if (B[k, col] == -1) {
            sum -= A[row, k];
        } 
      }

    output[row * N + col] = sum + bias[row];
  }
```

The code above corresponds to excution of a single thread. For more information on how CUDA parallelization is implemented in code and in reality, visit [here](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/).

### Optimizations

To see savings from our algorithm, we turn to weight packing algorithms. If we represent a weight in two bits, then we can take an eighth of the memory to store the weights, resulting in potentially significant speedups if memory loads are the limiting factor (as they often are). 

Because weight loads are significantly cheaper per capita than activation loads, we want to more heavily prioritize cache hits on activation loads. To do so, we can follow a number of strategies. Within each thread, we can compute a greater number of computations with the same activations. This may lead to speedups in memory loads, but significant optimization is required to find the balance between memory savings and parallelization costs. We can also use shared memory between threads, which allows for threads to compute different outputs in parallel, but without reloading activations each time. A barebones implementation is shown below:

```C++
  const uint col = blockIdx.x * blockDim.x + threadIdx.x;
  const uint row = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ half shared_activations[];

  if (row < M && col < N) {

    float sum = 0;

    for (int k = 0; k < K) {
      shared_activations[k] = A[row, k];
    }

    __syncthreads();

    for (int k = 0; k < K) {
      if (B[k, col] == 1) {
          sum += A[row, k];
      } else if (B[k, col] == -1) {
          sum -= A[row, k];
      } 
    }

    output[row * N + col] = sum + bias[row];
} 
```
The tradeoff with this implementation is the reliance on threads to be synchronized within the thread call, which is necessary to avoid race conditions but results in a performance penalty.

Further optimization is required to balance the size of K, M and N in scheduling the batch sizes to schedule to the block and individual threads. Because cuBLAS is highly optimized in this regard, we need a significant speedup to offset their gains. Later work can go into developing autotuning algorithms that dynamically schedule the computation as to maximize the L2 cache hit ratio and minimize synchronization delay. 