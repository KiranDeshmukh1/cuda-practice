#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)

extern "C" {
__global__ void sigmoid_kernel_f32(float *a, float *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // we need to lock the boundaries for f32 values since we using exponential
  if (idx < N) {
    float v = a[idx];
    v = fminf(fmaxf(v, MIN_EXP_F32), MAX_EXP_F32);
    y[idx] = 1.0f / (1.0f + expf(-v));
  }
}
}

extern "C" {
__global__ void sigmoid_kernel_f32x4(float *a, float *y, int N) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  // we need to lock the boundaries for f32 values since we using exponential
  if (idx < N) {
    float4 reg_v = *(reinterpret_cast<float4 *>(&(a[idx])));
    float4 reg_y;

    reg_v.x = fminf(fmaxf(reg_v.x, MIN_EXP_F32), MAX_EXP_F32);
    reg_v.y = fminf(fmaxf(reg_v.y, MIN_EXP_F32), MAX_EXP_F32);
    reg_v.z = fminf(fmaxf(reg_v.z, MIN_EXP_F32), MAX_EXP_F32);
    reg_v.w = fminf(fmaxf(reg_v.w, MIN_EXP_F32), MAX_EXP_F32);

    reg_y.x = 1.0f / (1.0f + expf(-reg_v.x));
    reg_y.y = 1.0f / (1.0f + expf(-reg_v.y));
    reg_y.z = 1.0f / (1.0f + expf(-reg_v.z));
    reg_y.w = 1.0f / (1.0f + expf(-reg_v.w));
    if (idx + 3 < N) {
      *(reinterpret_cast<float4 *>(&(y[idx]))) = reg_y;
    }
  }
}
}

extern "C" {
__global__ void sigmoid_kernel_f16(half *a, half *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // we need to lock the boundaries for f32 values since we using exponential
  const half f = __float2half(1.0f);
  if (idx < N) {
    half v = a[idx];
    v = __hmin(__hmax(v, MIN_EXP_F16), MAX_EXP_F16);
    y[idx] = f / (f + hexp(-v)); // this is valid in cuda 9.0+
  }
}
}

extern "C" {
__global__ void sigmoid_kernel_f16x2(half *a, half *y, int N) {
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx + 1 >= N) {
    return;
  }
  // we need to lock the boundaries for f32 values since we using exponential
  const half f = __float2half(1.0f);
  half2 reg_a = *(reinterpret_cast<half2 *>(&(a[idx])));
  half2 reg_y;
  reg_a.x = __hmin(__hmax(reg_a.x, MIN_EXP_F16), MAX_EXP_F16);
  reg_a.y = __hmin(__hmax(reg_a.y, MIN_EXP_F16), MAX_EXP_F16);

  reg_y.x = f / (f + hexp(-reg_a.x));
  reg_y.y = f / (f + hexp(-reg_a.y));
  *(reinterpret_cast<half2 *>(&(y[idx]))) = reg_y;
}
}

extern "C" {
__global__ void sigmoid_kernel_f16x8_pack(half *a, half *y, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx + 7 >= N) {
    return;
  }
  // we need to lock the boundaries for f32 values since we using exponential
  const half f = __float2half(1.0f);
  half reg_a[8];
  half reg_y[8];

  *reinterpret_cast<float4 *>(reg_a) = *reinterpret_cast<float4 *>(&a[idx]);

#pragma unroll
  for (int i = 0; i < 8; i++) {
    half v = __hmin(__hmax(reg_a[i], MIN_EXP_F16), MAX_EXP_F32);
    reg_y[i] = f / (f + hexp(-v));
  }

  *reinterpret_cast<float4 *>(&(y[idx])) = *reinterpret_cast<float4 *>(reg_y);
}
}
