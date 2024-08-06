#include <torch/extension.h>
#include <cuda_fp16.h>
#include "kernels.cuh"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>


#include <torch/extension.h>
#include <cuda_fp16.h>
#include <stdint.h>

void warptiling_bitlinear(int M, int N, int K, float alpha, torch::Tensor A, torch::Tensor B, float beta, torch::Tensor C) {

  // Settings for A6000
  const uint NUM_THREADS = 128;
  const uint BN = 128;
  const uint BM = 128;
  const uint BK = 16;
  const uint WN = 64;
  const uint WM = 64;
  const uint WNITER = 4;
  const uint TN = 4;
  const uint TM = 8;
  dim3 blockDim(NUM_THREADS);

  constexpr uint NUM_WARPS = NUM_THREADS / 32;

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

  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

  sgemmWarptiling_bitlinear<BM, BN, BK, WM, WN, WNITER, TM,
                  TN, NUM_THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A.data_ptr<half>(), B.data_ptr<uint8_t>(), beta, C.data_ptr<half>());
}

void warptiling(int M, int N, int K, float alpha, torch::Tensor A, torch::Tensor B, float beta, torch::Tensor C) {

  // Settings for A6000
  const uint NUM_THREADS = 128;
  const uint BN = 128;
  const uint BM = 128;
  const uint BK = 16;
  const uint WN = 64;
  const uint WM = 64;
  const uint WNITER = 4;
  const uint TN = 4;
  const uint TM = 8;
  dim3 blockDim(NUM_THREADS);

  constexpr uint NUM_WARPS = NUM_THREADS / 32;

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

  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

  sgemmWarptiling<BM, BN, BK, WM, WN, WNITER, TM,
                  TN, NUM_THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A.data_ptr<half>(), B.data_ptr<half>(), beta, C.data_ptr<half>());
}

PYBIND11_MODULE(warptiling, m) {
  m.def("warptiling_linear", &warptiling, "Run SGEMM Warptiling with Half Precision (CUDA)");
  m.def("warptiling_bitlinear", &warptiling_bitlinear, "Run SGEMM Warptiling with Half Precision (CUDA) and packed weights");
}