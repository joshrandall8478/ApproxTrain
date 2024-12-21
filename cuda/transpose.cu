#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "tensorflow/core/framework/types.h"
using namespace tensorflow;

template <typename T>
__global__ void transpose(const T* in, T* out, int rows, int cols) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;  // row index
    int c = blockIdx.x * blockDim.x + threadIdx.x;  // column index

    if (r < rows && c < cols)
    {
        int inIndex  = r * cols + c;      // row-major index in the original
        int outIndex = c * rows + r;      // transposed index
        out[outIndex] = in[inIndex];
    }
}



template __global__ void transpose<float>(const float* in, float* out, int rows, int cols);
template __global__ void transpose<int32>(const int32* in, int32* out, int rows, int cols);