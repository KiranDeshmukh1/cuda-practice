#!/bin/bash

# Define the CUDA source file and target PTX file
SOURCE_FILE="histogram_kernel.cu"
PTX_FILE="histogram_kernel.ptx"

# Compile the CUDA kernel to PTX with the appropriate flags
nvcc -O3 --use_fast_math \
    -U__CUDA_NO_HALF_OPERATORS__ \
    -U__CUDA_NO_HALF_CONVERSIONS__ \
    -U__CUDA_NO_HALF2_OPERATORS__ \
    -U__CUDA_NO_BFLOAT16_CONVERSIONS__ \
    --ptx $SOURCE_FILE -o $PTX_FILE

# Check if the compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful: $PTX_FILE created."
else
    echo "Compilation failed."
fi

