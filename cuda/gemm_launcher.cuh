#ifndef GEMM_LAUNCHER_CUH
#define GEMM_LAUNCHER_CUH
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
//=============================================================================
//===============================GEMM KERNEL===================================
//=============================================================================
// GEMM launcher with following parameters
// d: GPU device
// m: number of rows in matrix A
// n: number of columns in matrix B
// k: number of columns in matrix A
// a: matrix A
// lda: leading dimension of matrix A
// b: matrix B
// ldb: leading dimension of matrix B
// c: matrix C
// ldc: leading dimension of matrix C
// dim3 blockSize: block size
// dim3 gridSize: grid size
// approx_mul_lut<GPUDevice>& mul_lut: LUT for multiplication
// FloatMode

template <typename T>
void GEMM_LAUNCHER(
    const GPUDevice &d,
    size_t m,
    size_t n,
    size_t k,
    const T* a,
    size_t lda,
    const T* b,
    size_t ldb,
    T* c,
    size_t ldc,
    dim3 blockSize,
    dim3 gridSize,
    approx_mul_lut<GPUDevice>& mul_lut,
    FloatMode mode,
    bool forward_pass,
    bool input_grad
){

    /* Note: FP8 quantization happens prior to GEMM operations, see cuda_kernel.cu */
    if (mul_lut.is_lut()){
        // using case for different float modes
        switch (mode){
            case FloatMode::FP8E5M2:  
                gemm<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                break;
            case FloatMode::FP8HYB:
                gemm<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                break;
            case FloatMode::FP16:
                // use gemm_5exp with lut for both forward and backward pass
                // TODO: add actual implementation
                gemm_5exp<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc, mul_lut.get_mant_mul_lut_text_(), mul_lut.get_mant_mask_(), mul_lut.get_a_shift_(), mul_lut.get_b_shift_(), mul_lut.get_mant_width_());
                break;
            case FloatMode::BF16:
                // use gemm with lut for both forward and backward pass
                // the gemm supports 8-bit exponents, mantissa from 0 to 7bits, where 7bit is the bfloat16
                gemm<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc, mul_lut.get_mant_mul_lut_text_(), mul_lut.get_mant_mask_(), mul_lut.get_a_shift_(), mul_lut.get_b_shift_(), mul_lut.get_mant_width_());
                break;
            case FloatMode::FP32:
                // use gemm no lut is supported, if you really want to use some approximation, a behavior model is required that is C/C++ based and can be used in the kernel
                gemm<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                break;
            default:
                break;
        }
    } else {
        // using case for different float modes
        // no lut all accurate
        switch (mode){
            case FloatMode::FP8E5M2:  
                gemm<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                break;
            case FloatMode::FP8HYB:
                gemm<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                break;
            case FloatMode::FP16:
                // use gemm_fp16 without lut for both forward and backward pass
                gemm_fp16<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                break;
            case FloatMode::BF16:
                // use gemm_bf16 without lut for both forward and backward pass
                gemm_bf16<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                break;
            case FloatMode::FP32:
                // use gemm without lut for both forward and backward pass
                gemm<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                break;
            default:
                break;
        }

    }
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
// initialize the template

// template void GEMM_LAUNCHER<float>(
//     const GPUDevice &d,
//     size_t m,
//     size_t n,
//     size_t k,
//     const float* a,
//     size_t lda,
//     const float* b,
//     size_t ldb,
//     float* c,
//     size_t ldc,
//     dim3 blockSize,
//     dim3 gridSize,
//     approx_mul_lut<GPUDevice>& mul_lut,
//     FloatMode mode,
//     bool forward_pass
// );
// // initialize the template
// template void GEMM_LAUNCHER<int32>(
//     const GPUDevice &d,
//     size_t m,
//     size_t n,
//     size_t k,
//     const int32* a,
//     size_t lda,
//     const int32* b,
//     size_t ldb,
//     int32* c,
//     size_t ldc,
//     dim3 blockSize,
//     dim3 gridSize,
//     approx_mul_lut<GPUDevice>& mul_lut,
//     FloatMode mode,
//     bool forward_pass
// );
#endif  // GEMM_LAUNCHER_CUH