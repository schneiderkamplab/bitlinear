#include <cassert>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>

// A600 paramters
const uint BM = 64; // The threadblock size for M dimension SMEM caching.
const uint BN = 64; // The threadblock size for N dimension SMEM caching.
const uint BK = 8; // The threadblock size for K dimension SMEM caching.
const uint TM = 8; // The per-thread tile size for M dimension.

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

/*
Helper Functions for Vectorized Loads
*/
typedef struct {
    half data[8];
} half8;

typedef struct {
    char2 data[1];
} int8_2;

__device__ void loadfromGMEM(
    int N, int K, 
    const half *A, const int16_t *W,
    half *sA, int8_t *sW, 
    int innerRowA, int innerColA,
    int innerRowB, int innerColB
    ) {
    
    // Get the index of the shared_memory
    int idx = ...;
    int memIdx = ...;

    // Load 8 half values (128 bits) from global memory
    const half8 tmp = __ldg(&A[memIdx]);

    // Store the 8 half values into shared memory
    sA[8 * idx] = __low2half(tmp.data[0]);
    sA[8 * idx + 1] = __high2half(tmp.data[0]);
    sA[8 * idx + 2] = __low2half(tmp.data[1]);
    sA[8 * idx + 3] = __high2half(tmp.data[1]);
    sA[8 * idx + 4] = __low2half(tmp.data[2]);
    sA[8 * idx + 5] = __high2half(tmp.data[2]);
    sA[8 * idx + 6] = __low2half(tmp.data[3]);
    sA[8 * idx + 7] = __high2half(tmp.data[3]);


    // Load 128 bits from the int16_t array (8 int16_t values)
    short8 int16Data = reinterpret_cast<const short8*>(W)[...];
    int16Result[idx] = int16Data;


    // Synchronize to ensure all threads have written their data
    __syncthreads();

}


namespace btm {




}



__global__ void sgemm1DBlocktiling(
    int M, int N, int K,
    const half *A, 
    const int8_t *W,
    const half *bias, 
    const half *scale,
    half *output
    ) {

  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // each warp will calculate 32*TM elements, with 32 being the columnar dim.
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  // allocate space for the current blocktile in SMEM
  __shared__ half As[BM * BK];
  __shared__ half Ws[BN * BK];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // todo: adjust this to each thread to load multiple entries and
  // better exploit the cache sizes
  assert(BM * BK == blockDim.x);
  assert(BN * BK == blockDim.x);
  
  const uint innerColA = threadIdx.x % BK; // warp-level GMEM coalescing
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColB = threadIdx.x % BN; // warp-level GMEM coalescing
  const uint innerRowB = threadIdx.x / BN;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM] = {0.0};

  // outer loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    // advance blocktile
    A += BK;
    B += BK * N;

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // we make the dotproduct loop the outside loop, which facilitates
      // reuse of the Bs entry, which we can cache in a tmp var.
      float tmpB = Bs[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    output[(threadRow * TM + resIdx) * N + threadCol] = threadResults[resIdx] + output[(threadRow * TM + resIdx) * N + threadCol];
  }

}

torch::Tensor linear(
    int M, int N, int K, 
    half *A, 
    int8_t *W ,
    half *bias,
    half *scale
    ) {

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);
    auto output = torch::zeros({M, N}, options);

    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / TM);

    sgemmDBlocktiling<<<gridDim, blockDim>>>(
        M, N, K, 
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()), 
        W.data_ptr<int8_t>(), 
        reinterpret_cast<const half*>(bias.data_ptr<at::Half>()), 
        reinterpret_cast<half*>(scale.data_ptr<at::Half>())
        reinterpret_cast<half*>(output.data_ptr<at::Half>())
        );

    return output;
}