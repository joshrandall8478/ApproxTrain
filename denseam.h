#ifndef DENSEAM_H_
#define DENSEAM_H_

#include <unsupported/Eigen/CXX11/Tensor>
#include "approx_mul_lut.h"
#include "floatmode.h"
template <typename Device, typename T>
struct DenseamFunctor {
    void operator()(const Device& d, const T* inputs, const T* weights, T* output,
            const int batch, const int units, const int input_width,
            approx_mul_lut<Device>& mul_lut, FloatMode mode, T* quant_input, T* quant_weight, AccumMode accum_mode, size_t trunk_size = 0, uint8_t e4m3_exponent_bias = 7, uint8_t e5m2_exponent_bias = 31
            );
};

template <typename Device, typename T>
struct DenseamWeightGradFunctor{
    void operator()(const Device& d, const T* input, const T* grads,
            T* output, const int batch, const int units, const int input_width,
            approx_mul_lut<Device>& mul_lut, FloatMode mode, T* quant_input, T* quant_grad, AccumMode accum_mode, T *input_T, size_t trunk_size = 0, uint8_t e4m3_exponent_bias = 7, uint8_t e5m2_exponent_bias = 31
            );
};
template <typename Device, typename T>
struct DenseamInputGradFunctor{
    void operator()(const Device& d, const T* weight, const T* grads,
            T* output, const int batch, const int units, const int input_width,
            approx_mul_lut<Device>& mul_lut, FloatMode mode, T* quant_weight, T* quant_grad, AccumMode accum_mode, T *weight_T, size_t trunk_size = 0, uint8_t e4m3_exponent_bias = 7, uint8_t e5m2_exponent_bias = 31
            ); 
};


#if GOOGLE_CUDA
template <typename T>
struct DenseamFunctor<Eigen::GpuDevice, T> {
    void operator()(const Eigen::GpuDevice& d, const T* inputs, const T* weights, 
            T* output, const int batch, const int units, const int input_width,
            approx_mul_lut<Eigen::GpuDevice>& mul_lut, FloatMode mode, T* quant_input, T* quant_weight, AccumMode accum_mode, size_t trunk_size = 0, uint8_t e4m3_exponent_bias = 7, uint8_t e5m2_exponent_bias = 31
            );
};

template <typename T>
struct DenseamWeightGradFunctor<Eigen::GpuDevice, T>{
    void operator()(const Eigen::GpuDevice& d, const T* input, const T* grads,
            T* output, const int batch, const int units, const int input_width,
            approx_mul_lut<Eigen::GpuDevice>& mul_lut, FloatMode mode, T* quant_input, T* quant_grad, AccumMode accum_mode, T *input_T, size_t trunk_size = 0, uint8_t e4m3_exponent_bias = 7, uint8_t e5m2_exponent_bias = 31
            );
};
template <typename T>
struct DenseamInputGradFunctor<Eigen::GpuDevice, T>{
    void operator()(const Eigen::GpuDevice& d, const T* weight, const T* grads,
            T* output, const int batch, const int units, const int input_width,
            approx_mul_lut<Eigen::GpuDevice>& mul_lut, FloatMode mode,  T* quant_weight, T* quant_grad, AccumMode accum_mode, T *weight_T, size_t trunk_size = 0, uint8_t e4m3_exponent_bias = 7, uint8_t e5m2_exponent_bias = 31
            ); 
};
#endif
#endif
