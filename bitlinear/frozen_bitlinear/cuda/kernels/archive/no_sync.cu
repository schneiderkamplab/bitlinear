#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h> 

namespace py = pybind11;

/*
*********************************************************************
function name: linear_matmul_cuda
description: dot product of two arbitrarily sized matrices.
parameters:
  input: Input of size m X k.
  weights: weight kernel of size n X k packed in int8. Contains 
    1 at bit0 if weight is 1 and 1 at bit1 if weight is -1
  bias: bias per output channel.
  m,k,n: sizes of matrices.
return: none
*********************************************************************
*/
//////////////////////////////////////////////////////////////
// This performs the computations in fp32 for higher precision
//////////////////////////////////////////////////////////////
__global__ void fp32_linear_matmul_cuda(
  const half *input,
  const int8_t *weights,
  const half *bias,
  half *output,
  const int m,
  const int k,
  const int n
)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Check if we are within bounds
  if (col < n && row < m) {  

    float sum = 0.0f;
    int start = col * k;
    int shift = (start << 1) % 8;

    int index;
    int8_t weight;

    // Iterate through the interior dimension
    for (int i = 0; i < k; i++) {
        
      index = (start + i) >> 2;
      weight = (weights[index] >> shift) & 0x03;

      if (weight == 1) {
        sum += __half2float(input[row * k + i]);
      } else if (weight == 2) {
        sum -= __half2float(input[row * k + i]);
      } 

      shift = (shift + 2) % 8;
    }

    // Store the result into the correct location in memory
    output[row * n + col] = __float2half(sum + __half2float(bias[row]));

  }
}
/*
*********************************************************************
function name: linear
description: linear layer that calls the matmul kernel.
parameters:
  input: Input of size m X k.
  weights: weight kernel of size n X k packed in int8. Contains 
    1 at bit0 if weight is 1 and 1 at bit1 if weight is -1
  bias: bias per output channel.
  m,k,n: sizes of matrices.
  row_block_size: the number of rows to compute in parallel.
  col_block_size: the number of columns to compute in parallel.
return:
  output: output of size m x n.
*********************************************************************
*/
torch::Tensor linear(
  torch::Tensor input,
  torch::Tensor weights,
  torch::Tensor bias,
  int m,
  int k,
  int n,
  const unsigned int row_block_size,
  const unsigned int col_block_size
) {
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);
    auto output = torch::zeros({m, n}, options);

    unsigned int grid_rows = (m + row_block_size - 1) / row_block_size;
    unsigned int grid_cols = (n + col_block_size - 1) / col_block_size;

    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(col_block_size, row_block_size);

    fp32_linear_matmul_cuda<<<dimGrid, dimBlock>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        weights.data_ptr<int8_t>(),
        reinterpret_cast<const half*>(bias.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        m, k, n
    );

    // Check for any errors launching the kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    return output;
}


// Binding to generate the .so file, to call from Python.
PYBIND11_MODULE(no_stream, m) {
    m.doc() = "Implementation of bitlinear forward linear in CUDA";
    m.def("linear", &linear, "bitlinear_forward (CUDA)");
}
