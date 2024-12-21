#ifndef TRANSPOSE_CUH
#define TRANSPOSE_CUH

#include <cuda.h>
#include <cuda_fp16.h>
#include "gemm.cuh"
#include "approx_mul_lut.h"
#include "floatmode.h"
#include <iostream>
#include "tensorflow/core/framework/types.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;

template <typename T>
__global__ void transpose(const T* in, T* out, int rows, int cols);

template <typename T>
void transpose_launcher(const GPUDevice &d, const T* in, T* out, int rows, int cols){
    dim3 threads(16, 16);
    dim3 blocks((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);
    transpose<T><<<blocks, threads, 0, d.stream()>>>(in, out, rows, cols);
    // check error
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}



#endif // TRANSPOSE_CUH
