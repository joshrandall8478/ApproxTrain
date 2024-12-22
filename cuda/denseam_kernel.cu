

#define EIGEN_USE_GPU

#include "gpu_kernel_helper.h"
#include "error.cuh"
#include "denseam.h"
#include "approx_mul_lut.h"
#include <cuda_fp16.h>
#include "fp8_conversion.cuh"
#include <iostream>
#include "quant.cuh"
#include "floatmode.h"
#include "accumulate.cuh"
#include "gemm_launcher.cuh"
#include "transpose.cuh"
using namespace tensorflow;
using GpuDevice = Eigen::GpuDevice;

// Functor for forward pass
template <typename T>
void DenseamFunctor<GpuDevice, T>::operator()(
        const GpuDevice& d, const T* inputs, const T* weights, T* output,
        const int batch, const int units, const int input_width,
        approx_mul_lut<GpuDevice>& mul_lut, FloatMode mode, T* quant_input, T* quant_weight, AccumMode accum_mode, size_t trunk_size, uint8_t e4m3_exponent_bias,
    uint8_t e5m2_exponent_bias
        )
{ 
    T* input_data = const_cast<T*>(inputs);
    T* weight_data = const_cast<T*>(weights);
    // quantize the inputs and weights
    const int input_size = batch * input_width;
    const int weight_size = input_width * units;
    if (mode == FloatMode::FP8HYB || mode == FloatMode::FP8E5M2) {
        switch (mode){
            case FloatMode::FP8E5M2:  
                quant_fp32_e5m2clipping_launcher<T>(d, inputs, quant_input, input_size, e5m2_exponent_bias);
                quant_fp32_e5m2clipping_launcher<T>(d, weights, quant_weight, weight_size, e5m2_exponent_bias);
                break;
            case FloatMode::FP8HYB:
                quant_fp32_e4m3clipping_launcher<T>(d, inputs, quant_input, input_size, e4m3_exponent_bias);
                quant_fp32_e4m3clipping_launcher<T>(d, weights, quant_weight, weight_size, e4m3_exponent_bias);
                break;
            default:
                break;
            break;
        }
        input_data = quant_input;
        weight_data = quant_weight;
    }
    // gemm
    const size_t m = batch;
    const size_t n = units;
    const size_t k = input_width;
    const size_t lda = k;
    const size_t ldb = n;
    const size_t ldc = n;
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);
    GEMM_LAUNCHER<T>(d, m, n, k, input_data, lda, weight_data, ldb, output, ldc, blockSize, gridSize, mul_lut, mode, false, false, accum_mode, trunk_size);
}

// Functor for weight gradients
template <typename T>
void DenseamWeightGradFunctor<GpuDevice, T>::operator()
    (const GpuDevice& d, const T* inputs, const T* grads,
            T* output, const int batch, const int units, const int input_width,
            approx_mul_lut<GpuDevice>& mul_lut, FloatMode mode, T* quant_input, T* quant_grad, AccumMode accum_mode, T *input_T,
    size_t trunk_size, uint8_t e4m3_exponent_bias, uint8_t e5m2_exponent_bias)
{
    // transpose the input
    transpose_launcher<T>(d, inputs, input_T, batch, input_width);
    // quantize the inputs and grads
    const int input_size = batch * input_width;
    const int grad_size = batch * units;
    T* input_data = input_T;
    T* grad_data = const_cast<T*>(grads);
    if (mode == FloatMode::FP8HYB || mode == FloatMode::FP8E5M2) {
        switch (mode){
            case FloatMode::FP8E5M2:  
                quant_fp32_e5m2clipping_launcher<T>(d, input_T, quant_input, input_size, e5m2_exponent_bias);
                quant_fp32_e5m2clipping_launcher<T>(d, grads, quant_grad, grad_size, e5m2_exponent_bias);
                break;
            case FloatMode::FP8HYB:
                quant_fp32_e4m3clipping_launcher<T>(d, input_T, quant_input, input_size, e4m3_exponent_bias);
                quant_fp32_e5m2clipping_launcher<T>(d, grads, quant_grad, grad_size, e5m2_exponent_bias);
                break;
            default:
                break;
            break;
        }
        input_data = quant_input;
        grad_data = quant_grad;
    }
    // gemm
    const size_t m = input_width;
    const size_t n = units;
    const size_t k = batch;
    const size_t lda = k;
    const size_t ldb = n;
    const size_t ldc = n;
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);
    GEMM_LAUNCHER<T>(d, m, n, k, input_data, lda, grad_data, ldb, output, ldc, blockSize, gridSize, mul_lut, mode, true, false, accum_mode, trunk_size);
    
}

// Functor for input gradients
template <typename T>
void DenseamInputGradFunctor<GpuDevice, T>::operator()
    (const GpuDevice& d, const T* weights, const T* grads,
            T* output, const int batch, const int units, const int input_width,
            approx_mul_lut<GpuDevice>& mul_lut, FloatMode mode, T* quant_weight, T* quant_grad, AccumMode accum_mode, T* weight_T,
    size_t trunk_size, uint8_t e4m3_exponent_bias, uint8_t e5m2_exponent_bias)
{
    // transpose the weights
    transpose_launcher<T>(d, weights, weight_T, input_width, units);
    // quantize the weights and grads
    const int weight_size = input_width * units;
    const int grad_size = batch * units;
    T* weight_data = weight_T;
    T* grad_data = const_cast<T*>(grads);
    if (mode == FloatMode::FP8HYB || mode == FloatMode::FP8E5M2) {
        switch (mode){
            case FloatMode::FP8E5M2:  
                quant_fp32_e5m2clipping_launcher<T>(d, weight_T, quant_weight, weight_size, e5m2_exponent_bias);
                quant_fp32_e5m2clipping_launcher<T>(d, grads, quant_grad, grad_size, e5m2_exponent_bias);
                break;
            case FloatMode::FP8HYB:
                quant_fp32_e4m3clipping_launcher<T>(d, weight_T, quant_weight, weight_size, e4m3_exponent_bias);
                quant_fp32_e5m2clipping_launcher<T>(d, grads, quant_grad, grad_size, e5m2_exponent_bias);
                break;
            default:
                break;
            break;
        }
        weight_data = quant_weight;
        grad_data = quant_grad;
    }
    // gemm
    const size_t m = batch;
    const size_t n = input_width;
    const size_t k = units;
    const size_t lda = k;
    const size_t ldb = n;
    const size_t ldc = n;
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);
    GEMM_LAUNCHER<T>(d, m, n, k, grad_data, lda, weight_data, ldb, output, ldc, blockSize, gridSize, mul_lut, mode, false, true, accum_mode, trunk_size);
}

// Template instantiations
template struct DenseamFunctor<GpuDevice, float>;
template struct DenseamFunctor<GpuDevice, int32>;
template struct DenseamInputGradFunctor<GpuDevice, float>;
template struct DenseamInputGradFunctor<GpuDevice, int32>;
template struct DenseamWeightGradFunctor<GpuDevice, float>;
template struct DenseamWeightGradFunctor<GpuDevice, int32>;
