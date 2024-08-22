#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h> 

namespace py = pybind11;

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

__global__ void pack_weights_kernel(const half* weights, short* packed_weights, int n, int k) {
   
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int colStart = blockIdx.x * 8; 

    if (row < n && colStart < k) {
        
        short packed = 0;

        for (int index = 0; index < 8; ++index) {
            if (colStart + index < k) {
                float weight_value = __half2float(weights[row * k + colStart + index]);
                short bit_mask = (weight_value == 1.0f) ? (1 << (index * 2)) :
                                 (weight_value == -1.0f) ? (2 << (index * 2)) : 0;
                packed |= bit_mask;
            }
        }

        int packed_index = blockIdx.x * n + row; 
        packed_weights[packed_index] = packed;
    }
}

/*
In this implementation we have 8 weights in the K direction, which are then ordered in 
column major order ([0, 0-7], [1, 0-7], ... [(N-1), 0-7], [0, 8-15], ...) 
This comes from a NxK Weight Matrix 
*/
torch::Tensor packed_K8_column_major(torch::Tensor weights) {
    
    TORCH_CHECK(weights.is_contiguous(), "weights tensor must be contiguous");
    TORCH_CHECK(weights.dtype() == torch::kFloat16, "weights tensor must be of type float16");

    int n = weights.size(0);
    int k = weights.size(1);

    TORCH_CHECK(k % 8 == 0, "K must be divisible by 8");

    int packed_size = CEIL_DIV(n * k, 8);
    auto packed_weights = torch::zeros({packed_size}, torch::TensorOptions().dtype(torch::kInt16).device(torch::kCUDA));

    const unsigned int block_size = 128;
    const unsigned int grid_rows = CEIL_DIV(n, block_size);
    const unsigned int grid_cols = CEIL_DIV(k, 8);

    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(1, block_size);

    pack_weights_kernel<<<dimGrid, dimBlock>>>(
        reinterpret_cast<const half*>(weights.data_ptr<at::Half>()),
        packed_weights.data_ptr<short>(),
        n,
        k
    );

    cudaError_t err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed with error: ", cudaGetErrorString(err));

    return packed_weights;
}


PYBIND11_MODULE(pack_weights, m) {
    m.def("packed_K8_column_major", &packed_K8_column_major, "Pack fp16 weights into int16 tensor");
}