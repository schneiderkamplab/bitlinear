#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <stdint.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
const int WARPSIZE = 32; // warpSize is not constexpr

namespace wt {
template <const int BM, const int BN, const int BK, const int rowStrideA, const int rowStrideB>
__device__ void loadFromGmem(int N, int K, const half *A, const uint8_t *B, half *As, uint8_t *Bs, int innerRowA, int innerColA, int innerRowB, int innerColB) {
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    const float4 tmp = reinterpret_cast<const float4 *>(&A[(innerRowA + offset) * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA + offset] = __float2half(tmp.x);
    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = __float2half(tmp.y);
    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = __float2half(tmp.z);
    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = __float2half(tmp.w);
  }

  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    reinterpret_cast<uint8_t *>(&Bs[(innerRowB + offset) * BN + innerColB])[0] =
        reinterpret_cast<const uint8_t *>(&B[(innerRowB + offset) * (N / 4) + innerColB])[0];
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void processFromSmem(half *regM, uint8_t *regN, float *threadResults, const half *As, const uint8_t *Bs, const uint warpRow, const uint warpCol, const uint threadRowInWarp, const uint threadColInWarp) {
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // populate registers for whole warptile
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[wSubRowIdx * TM + i] = As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + i];
      }
    }
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      for (uint i = 0; i < TN; ++i) {
        regN[wSubColIdx * TN + i] = Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + i];
      }
    }

    // execute warptile matmul
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        // calculate per-thread results
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
          for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            // Unpack and process 2-bit weights
            uint8_t weight = (regN[wSubColIdx * TN + resIdxN / 4] >> ((resIdxN % 4) * 2)) & 0x03;
            float val = __half2float(regM[wSubRowIdx * TM + resIdxM]);
            if (weight == 1) {
              threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) + (wSubColIdx * TN) + resIdxN] += val;
            } else if (weight == 2) {
              threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) + (wSubColIdx * TN) + resIdxN] -= val;
            }
          }
        }
      }
    }
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int TM, const int TN,
          const int NUM_THREADS>
__global__ void sgemmWarptiling_bitlinear(const int M, const int N, const int K, const float alpha, const half *A, const uint8_t *B, const float beta, half *C) {
  const int tid = threadIdx.x;
  const int warpId = tid / 32;
  const int laneId = tid % 32;

  const int warpRow = warpId / (BN / WN);
  const int warpCol = warpId % (BN / WN);
  const int threadRowInWarp = laneId / (TN / 4);
  const int threadColInWarp = laneId % (TN / 4);

  __shared__ half As[BM * BK];
  __shared__ uint8_t Bs[BK * BN];

  float threadResults[TM * TN] = {0};

  for (int i = 0; i < K; i += BK) {
    wt::loadFromGmem<BM, BN, BK, 4, 1>(N, K, A, B, As, Bs, warpRow, i, warpCol, i);
    __syncthreads();

    wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, 1, 4, TM, TN>(threadResults, As, Bs, warpRow, warpCol, threadRowInWarp, threadColInWarp);
    __syncthreads();
  }

  for (int i = 0; i < TM; ++i) {
    for (int j = 0; j < TN; ++j) {
      const int globalRow = blockIdx.y * BM + warpRow * WM + threadRowInWarp * TM + i;
      const int globalCol = blockIdx.x * BN + warpCol * WN + threadColInWarp * TN + j;
      if (globalRow < M && globalCol < N) {
        C[globalRow * N + globalCol] = __float2half(threadResults[i * TN + j] + __half2float(C[globalRow * N + globalCol]));
      }
    }
  }
}
}