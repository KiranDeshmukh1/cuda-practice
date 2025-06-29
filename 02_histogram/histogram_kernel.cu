#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {
__global__ void histogram_i32_kernel(int *a, int *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    atomicAdd(&(y[a[idx]]), 1);
  }
}
}

extern "C" {
__global__ void histogram_i32x4_kernel(int *a, int *y, int N) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    int4 reg_a = *(reinterpret_cast<int4 *>(&(a[idx])));
    atomicAdd(&y[reg_a.x], 1);
    atomicAdd(&y[reg_a.y], 1);
    atomicAdd(&y[reg_a.z], 1);
    atomicAdd(&y[reg_a.w], 1);
  }
}
}
