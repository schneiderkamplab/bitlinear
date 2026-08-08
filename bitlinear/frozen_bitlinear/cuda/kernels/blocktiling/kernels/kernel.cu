#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

const uint NUM_THREADS = 256; // Number of Threads per block
const uint WARP_SIZE = 32; // Fixed in hardware
const uint NUM_WARPS = NUM_THREADS/WARP_SIZE; // 8

const uint TN = 8; // the number of elements calculated in the n dimension per thread
const uint TM = 8; // the number of elements calculated in the m dimension per thread loop

// Each warp will access 256 elements of A each in batches of 8
const uint STRIDE_Ak = 8; // The stride for loading A (in K direction)
const uint STRIDE_Wn = 8; // The stride for loading W (in N direction)

const uint BN = 2048; // NUM_THREADS*STRIDE_Wk*8, the threadblock size for N dimension W SMEM caching.
const uint BK = 512; // The threadblock size for K dimension A SMEM caching. 
const uint BM = 4; // NUM_WARPS, The number of rows to calculate per block

const uint WEIGHTSper32 = 16;
const uint HALVESper32 = 2;

namespace MemAccess {
    typedef struct {
    const __half data[8]
  } half8; 

  __device__ void loadAs(
    int K, 
    const __half *A, half2 *As,
    int innerColA, int innerRowA
    ) {

      // Load a half8 from A 
      const half8 tmp = reinterpret_cast<const half8*>(&A[innerRowA * K + innerColA* STRIDE_Ak])[0];

      // Convert half8 data to half2, transpose it and store in As
      As[(innerColA * 4 + 0) * BM + innerRowA] = __halves2half2(tmp.data[1], tmp.data[0]);
      As[(innerColA * 4 + 1) * BM + innerRowA] = __halves2half2(tmp.data[3], tmp.data[2]);
      As[(innerColA * 4 + 2) * BM + innerRowA] = __halves2half2(tmp.data[5], tmp.data[4]);
      As[(innerColA * 4 + 3) * BM + innerRowA] = __halves2half2(tmp.data[7], tmp.data[6]);

  } 

  __device__ void loadWs(
    const short *W, short *Ws,
    int innerRowW
    ) {

      // Load an int4 from W 
      const short8 tmp = reinterpret_cast<const int4*>(W)[innerRowW*STRIDE_Wn];

      // Convert int4 data to int16_t and store in Ws
      Ws[innerRowW*STRIDE_Wn]     = tmp[0];
      Ws[innerRowW*STRIDE_Wn + 1] = tmp[1];
      Ws[innerRowW*STRIDE_Wn + 2] = tmp[2];
      Ws[innerRowW*STRIDE_Wn + 3] = tmp[3];
      Ws[innerRowW*STRIDE_Wn + 4] = tmp[4];
      Ws[innerRowW*STRIDE_Wn + 5] = tmp[5];
      Ws[innerRowW*STRIDE_Wn + 6] = tmp[6];
      Ws[innerRowW*STRIDE_Wn + 7] = tmp[7];

  } 

  __device__ void load_regA(
    const half2 *As, float *regA,
    int index
    ) {
      
      const half8 tmp = reinterpret_cast<const half8*>(&As[index*BM])[0];
      
      // loading registers with the 8x2 activations from As

      for (int i=0; i<TM*2; i++) {
        regA[i]  = __half2float(tmp.data[i]);
      }

  } 



}

__global__ void linear(
  int M, int N, int K, 
  __half *A, short *W, 
  __half *output
  ) {

  // Move blocktile to beginning of A's row and B's column
  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  // allocate space for the current blocktile in smem
  __shared__ half2 As[BM * BK];
  __shared__ int Bs[GK * BN];

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const uint WarpRowA = threadIdx.x / WARP_SIZE;
  const uint innerColA = threadIdx.x % WARP_SIZE;
  const uint innerRowW = threadIdx.x;

  // allocate thread-local cache for results in registerfile
  float threadResults[TN*TM] = {0.0}; // We update a TMxTN block per thread
  float regA[BM*2] = {0.0}; // We iterate in a BMx2 block per 
  short regW[TN] = {0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    
    // populate the SMEM cache for the block of shared As
    MemAccess::loadAs(int K, A, As, innerColA, WarpRowA)
    A += BK; // move BK columns to right

    for (uint innerBlock; innerBlock < BK; innerBlock+=STRIDE_Ak) {
      
      MemAccess::loadWs(W, Ws, innerRowW)
      W += N; // move to next column
      __syncthreads();

      for (uint )
      MemAccess::load_regW(Ws, regW)


    }

    
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
      // load C vector into registers
      float4 tmp = reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
      // perform GEMM update in reg
      tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;
      tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
      tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
      tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
      // write back
      reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
          tmp;
    }
  }
}

void runSgemmVectorize(Torch:Tensor *A, int *B,
                       float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 and N >= 128) {
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmVectorize<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    // this is a hacky solution to the underlying problem
    // of not having proper bounds checking in the kernel
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmVectorize<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
}


torch::Tensor linear(
    torch::Tensor *A, 
    torch::Tensor *W ,
    half *bias,
    half *scale
    ) {
    
    int M = A.size(0);
    int K = A.size(1);
    int N = W.size(0);

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