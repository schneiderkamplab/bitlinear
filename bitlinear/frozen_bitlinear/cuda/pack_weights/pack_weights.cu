#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h> 

namespace py = pybind11;

__global__ void pack_weights_kernel(const half* weights, int* packed_weights, int n, int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < k) {
        int weight_index = row * k + col;
        float weight_value = __half2float(weights[weight_index]);

        int packed_index = weight_index / 4; // 4 weights per int8
        int bit_index = 2*weight_index % 8; // each weight takes 2 bits
        
        int bit_mask = (weight_value == 1.0f) ? (1 << bit_index) :
                          (weight_value == -1.0f) ? (2 << bit_index) : 0;

        atomicOr(&packed_weights[packed_index], bit_mask); // this avoids race condition errors from parallelization

    }
}

__global__ void int32_to_int8_kernel(const int* packed_weights_int32, int8_t* packed_weights_int8, int packed_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < packed_size) {
        int packed_value = packed_weights_int32[i];
        packed_weights_int8[i] = static_cast<int8_t>(packed_value & 0xFF);
    }
}

torch::Tensor packedint8(
    torch::Tensor weights,
    int n,
    int k
    ) {
    
    TORCH_CHECK(weights.is_contiguous(), "weights tensor must be contiguous");
    TORCH_CHECK(weights.dtype() == torch::kFloat16, "weights tensor must be of type float16");

    // Calculate size for packed weights tensor
    int packed_size = (n * k + 3) / 4; // 4 weights per int8

    auto packed_weights_int32 = torch::zeros({packed_size}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto packed_weights_int8 = torch::empty({packed_size}, torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA));

    const unsigned int block_size = 32;
    const unsigned int grid_rows = (n + block_size - 1) / block_size;
    const unsigned int grid_cols = (k + block_size - 1) / block_size;

    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(block_size, block_size);
    unsigned int grid_size = (packed_size + block_size - 1) / block_size;

    pack_weights_kernel<<<dimGrid, dimBlock>>>(
        reinterpret_cast<const half*>(weights.data_ptr<at::Half>()),
        packed_weights_int32.data_ptr<int>(),
        n,
        k
    );
    int32_to_int8_kernel<<<grid_size, block_size>>>(
        packed_weights_int32.data_ptr<int>(),
        packed_weights_int8.data_ptr<int8_t>(),
        packed_size
    );

    cudaError_t err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed with error: ", cudaGetErrorString(err));

    return packed_weights_int8;
}


PYBIND11_MODULE(pack_weights, m) {
    m.def("packedint8", &packedint8, "Pack fp16 weights into int8 tensor");
}