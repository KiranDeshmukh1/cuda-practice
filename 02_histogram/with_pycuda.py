import pycuda.driver as cuda
import pycuda.autoinit  # Initializes CUDA context
import numpy as np
# from pycuda.compiler import SourceModule
# import pycuda.driver as driver


mod = cuda.module_from_file("histogram_kernel.ptx")
histogram_i32_kernel = mod.get_function("histogram_i32_kernel")
histogram_i32x4_kernel = mod.get_function("histogram_i32x4_kernel")


def benchmark(a, y, mod, threads_per_block):
    # values = np.arange(10)
    # a = np.repeat(values, repeats).astype(np.int32)
    # y = np.zeros(values.shape[0]).astype(np.int32)

    a_gpu = cuda.mem_alloc(a.nbytes)
    y_gpu = cuda.mem_alloc(y.nbytes)

    cuda.memcpy_htod(a_gpu, a)

    N = a.shape[0]
    blocks_per_grid = (N + (threads_per_block - 1)) // threads_per_block

    mod(
        a_gpu,
        y_gpu,
        np.int32(N),
        block=(threads_per_block, 1, 1),
        grid=(blocks_per_grid, 1, 1),
    )

    cuda.Context.synchronize()

    cuda.memcpy_dtoh(y, y_gpu)

    for i in range(y.shape[0]):
        print(f" {i} : {y[i]}")


values = np.arange(10)
repeats = 1000

a = np.repeat(values, repeats).astype(np.int32)
y = np.zeros(values.shape[0]).astype(np.int32)

threads_per_block = 256


print("### 🖥️ Example Output\n")
print("```")
print("-" * 85)
print(" " * 40 + "h_i32")
benchmark(a, y, histogram_i32_kernel, threads_per_block)
print("-" * 85)
print(" " * 40 + "h_i32x4")
benchmark(a, y, histogram_i32x4_kernel, threads_per_block)
print("```")
