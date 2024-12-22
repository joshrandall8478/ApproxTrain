#ifndef quant_cuh
#define quant_cuh
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
__global__ void quant_fp32_e4m3clipping(const T* src, T* dest, size_t size, uint8_t exponent_bias = 7);
template <typename T>
__global__ void quant_fp32_e5m2clipping(const T* src, T* dest, size_t size, uint8_t exponent_bias = 31);

template <typename T>
void quant_fp32_e4m3clipping_launcher(const GPUDevice &d, const T* src, T* dest, size_t size, uint8_t exponent_bias = 7){
    int threadsperblock = 256;
    int blockspergrid = (size + threadsperblock - 1) / threadsperblock;
    quant_fp32_e4m3clipping<T><<<blockspergrid, threadsperblock, 0, d.stream()>>>(src, dest, size, exponent_bias);
    // check error
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

template <typename T>
void quant_fp32_e5m2clipping_launcher(const GPUDevice &d, const T* src, T* dest, size_t size, uint8_t exponent_bias = 31){
    int threadsperblock = 256;
    int blockspergrid = (size + threadsperblock - 1) / threadsperblock;
    quant_fp32_e5m2clipping<T><<<blockspergrid, threadsperblock, 0, d.stream()>>>(src, dest, size, exponent_bias);
    // check error
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
#endif