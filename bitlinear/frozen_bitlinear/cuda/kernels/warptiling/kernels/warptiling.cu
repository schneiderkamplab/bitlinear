#include <cassert>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>

// Settings for A6000
const uint NUM_THREADS = 128;
const uint BN = 128; // The threadblock size for N dimension SMEM caching.
const uint BM = 128; // The threadblock size for M dimension SMEM caching.
const uint BK = 16; // The threadblock size for K dimension SMEM caching.
const uint WN = 64; // N dim of continuous tile computed by each warp
const uint WM = 64; // M dim of continuous tile computed by each warp
const uint WNITER = 4; // The number of subwarp tiling steps in N dimension.
const uint TN = 4; // The per-thread tile size for N dimension.
const uint TM = 8; // The per-thread tile size for M dimension.
const int WARPSIZE = 32; // warpSize is not constexpr
const uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER); // The number of subwarp tiling steps in M dimension.
const uint WSUBM = WM / WMITER;
const uint WSUBN = WN / WNITER;
const uint rowStrideA = (NUM_THREADS * 4) / BK;
const uint rowStrideB = NUM_THREADS / (BN / 4);
const uint NUM_WARPS = NUM_THREADS / WARPSIZE;

#define CEIL_DIV(x, y) ((x) + (y)-1) / (y) 

/*
Half4 Helper Functions
*/

struct half4 {
  half2 x, y; // Each half2 contains two half values
};

__device__ half4 loadHalf4(const half* address) {
  half4 result;
  result.x = *reinterpret_cast<const half2*>(address);
  result.y = *reinterpret_cast<const half2*>(address + 2);
  return result;
}


/*
Memory Operations
*/

namespace wt {

  /*
  Load from Global Memory 4 items at a time
  */

  __device__ void loadFromGmem(
    int N, int K, 
    const half *A, const int8_t *W,
    half *sA, int8_t *sW, 
    int innerRowA, int innerColA,
    int innerRowB, int innerColB
    ) 
  {
    // Load matrix A into shared memory sA
    for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
      const half4 tmp = loadHalf4(&A[(innerRowA + offset) * K + innerColA * 4]);
      sA[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x.x;
      sA[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.x.y;
      sA[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.y.x;
      sA[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.y.y;
    }

    // Load 4 items matrix W into shared memory sW as packed 2-bit values
    for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
      sW[(innerRowB + offset) * (BN / 4) + innerColB] = W[(innerRowB + offset) * (K / 4) + innerColB];
    }
  } 

  /*
  Load from Shared Memory and Perform Computation
  */

  __device__ void processFromSmem(
    half *regM, int8_t *regN, 
    float *threadResults, 
    const half *sA, const int8_t *sW, 
    const uint warpRow, const uint warpCol,
    const uint threadRowInWarp, const uint threadColInWarp
    ) 
  {
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      
      // Load sub-matrix of A into registers for a whole warptile
      for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint i = 0; i < TM; ++i) {
          regM[wSubRowIdx * TM + i] =
              sA[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
                threadRowInWarp * TM + i];
        }
      }

      // Unpack and load sub-matrix of sW into registers already translated
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        for (uint i = 0; i < TN; ++i) {
          uint index = (warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + i) >> 2;
          int8_t packedWeights = sW[dotIdx * (BN / 4) + index];
          uint shift = ((warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + i) & 0x03) * 2;
          regN[wSubColIdx * TN + i] = (packedWeights >> shift) & 0x03;
        }
      }

      // execute warptile matmul
      for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
          // per-thread results
          for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
            for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
              // loads the value and corresponiding weight
              float inputVal = __half2float(regM[wSubRowIdx * TM + resIdxM]);
              int8_t weight = regN[wSubColIdx * TN + resIdxN];
              // if 0b01, adds the activation
              if (weight == 1) {
                threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                              (wSubColIdx * TN) + resIdxN] += inputVal;
              } 
              // if 0b10, subtracts the activation
              else if (weight == 2) {
                threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                              (wSubColIdx * TN) + resIdxN] -= inputVal;
              }
            }
          }
        }
      }
    }
  }

}

/*
CUDA Kernel for WarpTiling with the Bitlinear Implementation
*/
__global__ void __launch_bounds__(NUM_THREADS) sgemmWarptiling(
  const int M, const int N, const int K, 
  const half *A, const int8_t *W, half *C
  ) 
{
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // Placement of the warp in the threadblock tile
  const uint warpIdx = threadIdx.x / WARPSIZE; // the warp this thread is in
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  // Placement of the thread in the warp subtile
  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

  // allocate space for the current blocktile in SMEM
  __shared__ half sA[BM * BK];
  __shared__ int8_t sW[BK * BN / 4];

  // Move blocktile to beginning of A's row and W's column
  A += cRow * BM * K;
  W += cCol * BN / 4;
  // Move C_ptr to warp's output tile
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  // calculating the indices that this thread will load into SMEM
  // we'll load 4 elements per thread at each step
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
  // we cache into registers on the warptile level
  half regM[WMITER * TM] = {__float2half(0.0f)};
  int8_t regN[WNITER * TN] = {0};

  // loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    
    wt::loadFromGmem(N, K, A, W, sA, sW, innerRowA, innerColA, innerRowB, innerColB);
    
    __syncthreads();
    
    wt::processFromSmem(regM, regN, threadResults, sA, sW, warpRow, warpCol, threadRowInWarp, threadColInWarp);
    
    A += BK;
    W += BK * N / 4;

    __syncthreads();
  }

  // write out the results
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      // move C pointer to current warp subtile
      half *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 2) {
          // load C vector into registers
          half2 tmp = reinterpret_cast<half2 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0];
          // perform GEMM update in reg
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
          tmp.x = __float2half(threadResults[i + 0] + __half2float(tmp.x));
          tmp.y = __float2half(threadResults[i + 1] + __half2float(tmp.y));
          // write back
          reinterpret_cast<half2 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0] = tmp;
        }
      }
    }
  }
}


/*
Callable Function that computes the "matmul" of A and W and stores in C
*/
void matmul(int M, int N, int K, torch::Tensor A, torch::Tensor W, torch::Tensor C) {

  // warptile in threadblocktile
  static_assert((BN % WN == 0) and (BM % WM == 0));
  static_assert((BN / WN) * (BM / WM) == NUM_WARPS);

  // threads in warpsubtile
  static_assert((WM * WN) % (WARPSIZE * TM * TN * WNITER) == 0);
  constexpr uint WMITER = (WM * WN) / (32 * TM * TN * WNITER);
  
  // warpsubtile in warptile
  static_assert((WM % WMITER == 0) and (WN % WNITER == 0));

  static_assert((NUM_THREADS * 4) % BK == 0,
                "NUM_THREADS*4 must be multiple of BK to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of Bs during each iteration)");
  static_assert((NUM_THREADS * 4) % BN == 0,
                "NUM_THREADS*4 must be multiple of BN to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of As during each iteration)");
  static_assert(BN % (16 * TN) == 0,
                "BN must be a multiple of 16*TN to avoid quantization effects");
  static_assert(BM % (16 * TM) == 0,
                "BM must be a multiple of 16*TM to avoid quantization effects");
  static_assert((BM * BK) % (4 * NUM_THREADS) == 0,
                "BM*BK must be a multiple of 4*256 to vectorize loads");
  static_assert((BN * BK) % (4 * NUM_THREADS) == 0,
                "BN*BK must be a multiple of 4*256 to vectorize loads");

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

  sgemmWarptiling<<<gridDim, blockDim>>>(
    M, N, K, 
    reinterpret_cast<const half*>(A.data_ptr<at::Half>()), 
    W.data_ptr<int8_t>(), 
    reinterpret_cast<half*>(C.data_ptr<at::Half>())
    );
}

PYBIND11_MODULE(warptiling, m) {
  m.def("matmul", &matmul, "Run SGEMM Warptiling with Half Precision (CUDA)");
}