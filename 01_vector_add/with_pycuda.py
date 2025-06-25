import pycuda.driver as cuda
import pycuda.autoinit  # Initializes CUDA context
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.driver as driver
import time
import random
import torch
from typing import Optional


def benchmark(
    module: callable,
    a,
    b,
    c,
    N,
    threads_per_block,
    tag: str,
    warmup: int = 10,
    iters: int = 1000,
):
    a_gpu = cuda.mem_alloc(a.nbytes)
    b_gpu = cuda.mem_alloc(b.nbytes)
    c_gpu = cuda.mem_alloc(c.nbytes)
    
    cuda.memcpy_htod(a_gpu,a)
    cuda.memcpy_htod(b_gpu,b)


    blocks_per_grid = (N + (threads_per_block - 1)) // threads_per_block
        # warmup
    for i in range(warmup):
        module(
            a_gpu,
            b_gpu,
            c_gpu,
            np.int32(N),
            block=(threads_per_block,1,1),
            grid=(blocks_per_grid,1,1),
        )

    # iters
    start_event = driver.Event()
    end_event = driver.Event()
    start_event.record()

    for i in range(iters):
        module(
            a_gpu,
            b_gpu,
            c_gpu,
            np.int32(N),
            block=(threads_per_block,1,1),
            grid=(blocks_per_grid,1,1),
        )

    end_event.record()
    end_event.synchronize()
    
    cuda.memcpy_dtoh(c, c_gpu)

    total_time = end_event.time_since(start_event)
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = c.copy()[:2].tolist()
    out_val = [round(v, 8) for v in out_val]
    print(f"{out_info:>18}: {out_val}, time:{mean_time:.8f} ms")


mod = cuda.module_from_file("vector_add.ptx")
vector_add_f16 = mod.get_function("add_kernel_f16")
vector_add_f32 = mod.get_function("add_kernel_f32")
vector_add_f32x4 = mod.get_function("add_kernel_f32x4")
vector_add_f16x2 = mod.get_function("add_kernel_f16x2")

vector_add_f16x8 = mod.get_function("add_kernel_f16x8")

vector_add_f16x8_pack = mod.get_function("add_kernel_f16x8_pack")

N = 1024 * 1024
a = np.random.rand(N).astype(np.float32)
b = np.random.rand(N).astype(np.float32)
c = np.zeros_like(a)

d = np.random.rand(N).astype(np.float16)
e = np.random.rand(N).astype(np.float16)
f = np.zeros_like(e)

threads_per_block = 256

print("-"* 85)
print(" "* 40 + f"N = {N}")
benchmark(vector_add_f32, a, b, c, N, threads_per_block, tag='f32')

benchmark(vector_add_f32x4, a, b, c, N, threads_per_block, tag='f32x4')

benchmark(vector_add_f16, d, e, f, N, threads_per_block, tag='f16')

benchmark(vector_add_f16x2, d, e, f, N, threads_per_block, tag='f16x2')

benchmark(vector_add_f16x8, d, e, f, N, threads_per_block, tag='f16x8')

benchmark(vector_add_f16x8_pack, d, e, f, N, threads_per_block, tag='f16x8_pack')


print("-"* 85)
