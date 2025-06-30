import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time


def benchmark(kernel, a, y, N, threads_per_block, tag):
    a_gpu = cuda.mem_alloc(a.nbytes)
    y_gpu = cuda.mem_alloc(y.nbytes)

    cuda.memcpy_htod(a_gpu, a)
    blocks_per_thread = (N + (threads_per_block - 1)) // threads_per_block

    # warm up
    for _ in range(10):
        kernel(
            a_gpu,
            y_gpu,
            np.int32(N),
            block=(threads_per_block, 1, 1),
            grid=(blocks_per_thread, 1, 1),
        )

    cuda.Context.synchronize()
    # actual
    start_time = time.time()
    for _ in range(1000):
        kernel(
            a_gpu,
            y_gpu,
            np.int32(N),
            block=(threads_per_block, 1, 1),
            grid=(blocks_per_thread, 1, 1),
        )

    cuda.Context.synchronize()

    end_time = time.time()
    cuda.memcpy_dtoh(y, y_gpu)
    total_time = (end_time - start_time) * 1000
    average_time = total_time / 1000
    print(" " * 65 + f"N: {N}")
    print(f"out-{tag} : {y[:2]}, time : {average_time:.8f} ms")
    print("*" * 5)


# ---------------------------------------------------------------
mod = cuda.module_from_file("sigmoid_kernel.ptx")
sigmoid_kernel_f32 = mod.get_function("sigmoid_kernel_f32")
sigmoid_kernel_f32x4 = mod.get_function("sigmoid_kernel_f32x4")
sigmoid_kernel_f16 = mod.get_function("sigmoid_kernel_f16")
sigmoid_kernel_f16x2 = mod.get_function("sigmoid_kernel_f16x2")
sigmoid_kernel_f16x8_pack = mod.get_function("sigmoid_kernel_f16x8_pack")
threads_per_block = 256
Ns = (1024 * 1024, 4096 * 4096)
for N in Ns:
    a = np.random.rand(N).astype(np.float32)
    y = np.zeros_like(a)
    b = np.random.rand(N).astype(np.float16)
    z = np.zeros_like(b)

    benchmark(sigmoid_kernel_f32, a, y, N, threads_per_block, "f32")
    benchmark(sigmoid_kernel_f32x4, a, y, N, threads_per_block, "f32x4")
    benchmark(sigmoid_kernel_f16, b, z, N, threads_per_block, "f16")
    benchmark(sigmoid_kernel_f16x2, b, z, N, threads_per_block, "f16x2")
    benchmark(sigmoid_kernel_f16x8_pack, b, z, N, threads_per_block, "f16x8_pack")
    print("-" * 85)
