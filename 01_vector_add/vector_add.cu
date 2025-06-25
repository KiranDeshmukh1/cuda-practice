#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

extern "C" {

__global__ void add_kernel_f32(const float *a, const float *b, float *c,
                               int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    c[idx] = a[idx] + b[idx];
  }
}
}

extern "C" {

// CUDA kernel for elementwise addition
__global__ void add_kernel_f32x4(float *a, float *b, float *c, int N) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    float4 reg_a = FLOAT4(a[idx]);
    float4 reg_b = FLOAT4(b[idx]);
    float4 reg_c;

    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;

    FLOAT4(c[idx]) = reg_c;
  }
}
}

extern "C" {

__global__ void add_kernel_f16(half *a, half *b, half *c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    c[idx] = __hadd(a[idx], b[idx]);
  }
}
}

extern "C" {

__global__ void add_kernel_f16x2(half *a, half *b, half *c, int N) {
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    half2 reg_a = HALF2(a[idx]);
    half2 reg_b = HALF2(b[idx]);
    half2 reg_c;

    reg_c.x = __hadd(reg_a.x, reg_b.x);
    reg_c.y = __hadd(reg_a.y, reg_b.y);

    HALF2(c[idx]) = reg_c;
  }
}
}

extern "C" {

__global__ void add_kernel_f16x8(half *a, half *b, half *c, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    half2 reg_a_0 = HALF2(a[idx + 0]);
    half2 reg_a_1 = HALF2(a[idx + 2]);
    half2 reg_a_2 = HALF2(a[idx + 4]);
    half2 reg_a_3 = HALF2(a[idx + 6]);

    half2 reg_b_0 = HALF2(b[idx + 0]);
    half2 reg_b_1 = HALF2(b[idx + 2]);
    half2 reg_b_2 = HALF2(b[idx + 4]);
    half2 reg_b_3 = HALF2(b[idx + 6]);

    half2 reg_c_0, reg_c_1, reg_c_2, reg_c_3;

    reg_c_0.x = __hadd(reg_a_0.x, reg_b_0.x);
    reg_c_0.y = __hadd(reg_a_0.y, reg_b_0.y);
    reg_c_1.x = __hadd(reg_a_1.x, reg_b_1.x);
    reg_c_1.y = __hadd(reg_a_1.y, reg_b_1.y);
    reg_c_2.x = __hadd(reg_a_2.x, reg_b_2.x);
    reg_c_2.y = __hadd(reg_a_2.y, reg_b_2.y);
    reg_c_3.x = __hadd(reg_a_3.x, reg_b_3.x);
    reg_c_3.y = __hadd(reg_a_3.y, reg_b_3.y);

    if ((idx + 0) < N) {
      HALF2(c[idx + 0]) = reg_c_0;
    }
    if ((idx + 2) < N) {
      HALF2(c[idx + 2]) = reg_c_1;
    }
    if ((idx + 4) < N) {
      HALF2(c[idx + 4]) = reg_c_2;
    }
    if ((idx + 6) < N) {
      HALF2(c[idx + 6]) = reg_c_3;
    }
  }
}
}

extern "C" {

__global__ void add_kernel_f16x8_pack(half *a, half *b, half *c, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);

  if ((idx + 7) < N) {

    half2 pack_a[4], pack_b[4], pack_c[4];

    *(reinterpret_cast<float4 *>(pack_a)) =
        *(reinterpret_cast<float4 *>(&a[idx]));
    *(reinterpret_cast<float4 *>(pack_b)) =
        *(reinterpret_cast<float4 *>(&b[idx]));

#pragma unroll
    for (int i = 0; i < 4; i += 2) {
      HALF2(pack_c[i]) = __hadd2(pack_a[i], pack_b[i]);
    }

    *(reinterpret_cast<float4 *>(&c[idx])) =
        *(reinterpret_cast<float4 *>(pack_c));
  }
}
}
