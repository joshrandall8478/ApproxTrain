#include "tensorflow/core/framework/types.h"
#include "fp8_conversion.cuh"
using namespace tensorflow;


template <typename T>
__global__ void quant_fp32_e4m3clipping(const T* src, T* dest, size_t size, uint8_t exponent_bias = 7){
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < size){
        dest[index] = clip_fp8_e4m3(src[index], exponent_bias);
    }
}

template <typename T>
__global__ void quant_fp32_e5m2clipping(const T* src, T* dest, size_t size, uint8_t exponent_bias = 15){
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < size){
        dest[index] = clip_fp8_e5m2(src[index], exponent_bias);
    }
}

template __global__ void quant_fp32_e4m3clipping<float>(const float* src, float* dest, size_t size, uint8_t exponent_bias);
template __global__ void quant_fp32_e5m2clipping<float>(const float* src, float* dest, size_t size, uint8_t exponent_bias);
template __global__ void quant_fp32_e4m3clipping<int32>(const int32* src, int32* dest, size_t size, uint8_t exponent_bias);
template __global__ void quant_fp32_e5m2clipping<int32>(const int32* src, int32* dest, size_t size, uint8_t exponent_bias);