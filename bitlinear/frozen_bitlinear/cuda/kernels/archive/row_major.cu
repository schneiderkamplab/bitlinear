#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h> 

#define CEIL_DIV(x, y) ((x) + (y)-1) / (y) 

__global__ void naive_kernel(
    const half *input,
    const int8_t *weights,
    const half *bias,
    half *output, 
    int M, 
    int N,
    int K 
    ) 
{
  const uint col = blockIdx.x * blockDim.x + threadIdx.x;
  const uint row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < M && col < N) {

    float sum = 0.0f;
    int8_t weight;

     for (int k = 0; k < K; k += 4) {
      weight = weights[(col * K + k) >> 2];
    
      for (int offset=0; offset<4; offset++) {
        int8_t mask = (weight & (3 << (2 * offset))) >> (2 * offset);

        float input_val = __half2float(input[row * K + k + offset]);

        if (mask == 1) {
            sum += input_val;
        } else if (mask == 2) {
            sum -= input_val;
        } 
      }

    }

    // Store the result into the correct location in memory
    output[row * N + col] = __float2half(sum + __half2float(bias[row]));
    
  }
}

torch::Tensor linear(
  torch::Tensor input,
  torch::Tensor weights,
  torch::Tensor bias,
  int M,
  int N,
  int K
) {
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);
    auto output = torch::zeros({M, N}, options);

    uint blockSize = 32;

    dim3 dimGrid(CEIL_DIV(N, blockSize), CEIL_DIV(M, blockSize));
    dim3 dimBlock(blockSize, blockSize);

    naive_kernel<<<dimGrid, dimBlock>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        weights.data_ptr<int8_t>(),
        reinterpret_cast<const half*>(bias.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        M, N, K
    );

    return output;
}


// Binding to generate the .so file, to call from Python.
PYBIND11_MODULE(naive, m) {
    m.doc() = "Implementation of bitlinear forward linear in CUDA";
    m.def("linear", &linear, "bitlinear_forward (CUDA)");
}

void run_sgemm_shared_mem_block(int M, int N, int K, float alpha, float *A,
                                float *B, float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32 * 32);
  // L1 cache becomes useless, since we access GMEM only via SMEM, so we carve
  // out all of L1 to SMEM. This doesn't currently make a difference, since
  // occupancy is limited by reg and thread count, but it's good to do anyway.
  cudaFuncSetAttribute(sgemm_shared_mem_block<32>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  sgemm_shared_mem_block<32>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

__global__ void shared_memory(
    const half *input,
    const int8_t *weights,
    const half *bias,
    half *output, 
    int M, 
    int N,
    int K,
    int computeBlockSize,
    int blockSize
    ) 
{
  const uint blockRow = blockIdx.x;
  const uint threadCol = blockSize * blockIdx.y + threadIdx.x * computeBlockSize;

  // one row shared between all threads in a block
  __shared__ half shared_input[K];

  // advance pointers to the starting positions
  input += blockRow * K;    
  weights += threadCol * K;   
  output += blockRow * N + threadCol;

  float sum = 0.0;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // Have each thread load one of the elements in A & B
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    shared_input[blockRow * BLOCKSIZE + threadCol] = A[blockRow * K + threadCol];
    Bs[blockRow * BLOCKSIZE + threadCol] = B[blockRow * N + threadCol];

    // block threads in this block until cache is fully populated
    __syncthreads();

    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    // execute the dotproduct on the currently cached block
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] *
             Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
  }
  C[threadRow * N + threadCol] =
      alpha * tmp + beta * C[threadRow * N + threadCol];
}