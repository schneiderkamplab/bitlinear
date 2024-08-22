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

__global__ void pack_weights_kernel(const __half* weights, int* packed_weights, int n, int k) {
   
    int rowFactor = blockIdx.y * blockDim.y + threadIdx.y
    int row = rowFactor * 2;
    int colStart = blockIdx.x * 8; 

    if (row < n && colStart < k) {
        
        int packed = 0;

        for (innerRow=0, innerRow<2, ++innerRow) {
        
            for (int index=0; index<8; ++index) {

                if (colStart + index < k && innerRow + row < n) {

                    float weight_value = __half2float(weights[(row + innerRow) * k + colStart + index]);
                    int bit_mask = (weight_value == 1.0f) ? (1 << (innerRow*16 + index*2)) :
                                    (weight_value == -1.0f) ? (2 << (innerRow*16 + index*2)) : 0;
                    
                    packed |= bit_mask;
                }
            }
        }

        int packed_index = blockIdx.x * n + row / 2; 
        packed_weights[packed_index] = packed;
    }
}

/*
In this implementation we have 16 weights in the K direction, which are then ordered in 
column major order ([0, 0-15], [1, 0-15], ... [(N-1), 0-15], [0, 16-31], ...) 
This comes from a NxK Weight Matrix 
*/
torch::Tensor packed_K16_row_major(torch::Tensor weights) {
    
    TORCH_CHECK(weights.is_contiguous(), "weights tensor must be contiguous");
    TORCH_CHECK(weights.dtype() == torch::kFloat16, "weights tensor must be of type float16");

    int n = weights.size(0);
    int k = weights.size(1);

    TORCH_CHECK(k % 8 == 0, "K must be divisible by 8");

    auto packed_weights = torch::zeros({n/2, k/8}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    const unsigned int block_size = 64;
    const unsigned int grid_rows = CEIL_DIV(n/2, block_size);
    const unsigned int grid_cols = CEIL_DIV(k, 8);

    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(1, block_size);

    pack_weights_kernel<<<dimGrid, dimBlock>>>(
        reinterpret_cast<const __half*>(weights.data_ptr<at::Half>()),
        packed_weights.data_ptr<int>(),
        n,
        k
    );

    cudaError_t err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed with error: ", cudaGetErrorString(err));

    return packed_weights;
}


PYBIND11_MODULE(pack_weights, m) {
    m.def("packed_K16_row_major", &packed_K16_row_major, "Pack fp16 weights into int32 tensor");
}