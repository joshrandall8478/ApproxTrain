

#define EIGEN_USE_GPU

#include "gpu_kernel_helper.h"
#include "error.cuh"
#include "denseam.h"
#include "approx_mul_lut.h"
#include <cuda_fp16.h>
#include "fp8_conversion.cuh"
#include <iostream>
using namespace tensorflow;
using GpuDevice = Eigen::GpuDevice;
// start of new kernels
#ifdef RTZ
    #define fp32_add(a,b) __fadd_rz((a), (b));
#else
    #define fp32_add(a,b) ((a)+(b));
#endif

// non-lut fp32
template <typename T>
__global__ void DenseamKernel(
    const T* inputs,
    const T* weights,
    const int batch, 
    const int units, 
    const int input_width, 
    T* output
    ) 
{ 
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; 
    if(ix < units*batch)
    {
        int ix_unit = ix % units ;
        int ix_sample = ix / units;
        output[ix] = T(0);
        for (int ix_input = 0; ix_input < input_width; ix_input++)
        {
          T mul = inputs[ix_sample*input_width+ix_input] * weights[ix_input*units+ix_unit];
          output[ix] = fp32_add(mul, output[ix]);
        }  
    }
};
template <typename T>
__global__ void DenseamWeightsKernel(
    const T* grads,
    const T* inputs,
    const int input_width, 
    const int batch, 
    const int units, 
    T* grad_weights
    ) 
{ 
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; 
    if(ix < units*input_width)
    {
        int ix_unit = ix % units ;
        int ix_input = ix / units;
        grad_weights[ix] = T(0);
        for (int ix_sample = 0; ix_sample < batch; ix_sample++)
        {
            T mul = inputs[input_width*ix_sample+ix_input] * grads[ix_sample*units+ix_unit];
            grad_weights[ix] = fp32_add(mul, grad_weights[ix]);
        }  
    }
};

template <typename T>
__global__ void DenseamInputKernel(
    const T* grads,
    const T* weights,
    const int input_width, 
    const int batch, 
    const int units, 
    T* grad_inputs
    ) 
{ 
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; 
    if(ix < batch *input_width)
    {
        int ix_input = ix % input_width;
        int ix_sample = ix / input_width ;
        grad_inputs[ix] = T(0);

        for (int ix_unit = 0; ix_unit < units; ix_unit++)
        {   
            T mul = weights[ix_input*units+ ix_unit] * grads[ix_sample*units+ix_unit];
            grad_inputs[ix] = fp32_add(mul, grad_inputs[ix]);
        }
    }
};
// non-lut bf16
// clip the values to bf16 then convert it back to fp32 (truncation)
__device__ __forceinline__ float clip_bf16(float a) {
    return __uint_as_float(__float_as_uint(a) & 0xffff0000);
}
template <typename T>
__global__ void DenseamKernel_bf16(
    const T* inputs,
    const T* weights,
    const int batch, 
    const int units, 
    const int input_width, 
    T* output
    ) 
{ 
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; 
    if(ix < units*batch)
    {
        int ix_unit = ix % units ;
        int ix_sample = ix / units;
        output[ix] = T(0);
        for (int ix_input = 0; ix_input < input_width; ix_input++)
        {
          T mul = clip_bf16(inputs[ix_sample*input_width+ix_input]) * clip_bf16(weights[ix_input*units+ix_unit]);
          output[ix] = fp32_add(mul, output[ix]);
        }  
    }
};
template <typename T>
__global__ void DenseamWeightsKernel_bf16(
    const T* grads,
    const T* inputs,
    const int input_width, 
    const int batch, 
    const int units, 
    T* grad_weights
    ) 
{ 
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; 
    if(ix < units*input_width)
    {
        int ix_unit = ix % units ;
        int ix_input = ix / units;
        grad_weights[ix] = T(0);
        for (int ix_sample = 0; ix_sample < batch; ix_sample++)
        {
            T mul = clip_bf16(inputs[input_width*ix_sample+ix_input]) * clip_bf16(grads[ix_sample*units+ix_unit]);
            grad_weights[ix] = fp32_add(mul, grad_weights[ix]);
        }  
    }
};
template <typename T>
__global__ void DenseamInputKernel_bf16(
    const T* grads,
    const T* weights,
    const int input_width, 
    const int batch, 
    const int units, 
    T* grad_inputs
    ) 
{ 
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; 
    if(ix < batch *input_width)
    {
        int ix_input = ix % input_width;
        int ix_sample = ix / input_width ;
        grad_inputs[ix] = T(0);

        for (int ix_unit = 0; ix_unit < units; ix_unit++)
        {   
            T mul = clip_bf16(weights[ix_input*units+ ix_unit]) * clip_bf16(grads[ix_sample*units+ix_unit]);
            grad_inputs[ix] = fp32_add(mul, grad_inputs[ix]);
        }
    }
};

// non-lut fp16
// clip the values to fp16 then convert it back to fp32 
__device__ __forceinline__ float clip_fp16(float a) {
    return __half2float(__float2half(a));
}
template <typename T>
__global__ void DenseamKernel_fp16(
    const T* inputs,
    const T* weights,
    const int batch, 
    const int units, 
    const int input_width, 
    T* output
    ) 
{ 
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; 
    if(ix < units*batch)
    {
        int ix_unit = ix % units ;
        int ix_sample = ix / units;
        output[ix] = T(0);
        for (int ix_input = 0; ix_input < input_width; ix_input++)
        {
          T mul = clip_fp16(inputs[ix_sample*input_width+ix_input]) * clip_fp16(weights[ix_input*units+ix_unit]);
          output[ix] = fp32_add(mul, output[ix]);
        }  
    }
};
template <typename T>
__global__ void DenseamWeightsKernel_fp16(
    const T* grads,
    const T* inputs,
    const int input_width, 
    const int batch, 
    const int units, 
    T* grad_weights
    ) 
{ 
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; 
    if(ix < units*input_width)
    {
        int ix_unit = ix % units ;
        int ix_input = ix / units;
        grad_weights[ix] = T(0);
        for (int ix_sample = 0; ix_sample < batch; ix_sample++)
        {
            T mul = clip_fp16(inputs[input_width*ix_sample+ix_input]) * clip_fp16(grads[ix_sample*units+ix_unit]);
            grad_weights[ix] = fp32_add(mul, grad_weights[ix]);
        }  
    }
};
template <typename T>
__global__ void DenseamInputKernel_fp16(
    const T* grads,
    const T* weights,
    const int input_width, 
    const int batch, 
    const int units, 
    T* grad_inputs
    ) 
{ 
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; 
    if(ix < batch *input_width)
    {
        int ix_input = ix % input_width;
        int ix_sample = ix / input_width ;
        grad_inputs[ix] = T(0);

        for (int ix_unit = 0; ix_unit < units; ix_unit++)
        {   
            T mul = clip_fp16(weights[ix_input*units+ ix_unit]) * clip_fp16(grads[ix_sample*units+ix_unit]);
            grad_inputs[ix] = fp32_add(mul, grad_inputs[ix]);
        }
    }
};
/* non-lut fp8*/
// clip the values to fp8 then convert it back to fp32 (truncation)
// fp32 to e4m3 then convert back to fp32
__device__ __forceinline__ float clip_fp8_e4m3(float a) {
    return e4m3_to_fp32(fp32_to_e4m3(a));
}
template <typename T>
__global__ void DenseamKernel_fp8_e4m3(
    const T* inputs,
    const T* weights,
    const int batch, 
    const int units, 
    const int input_width, 
    T* output
    ) 
{ 
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; 
    if(ix < units*batch)
    {
        int ix_unit = ix % units ;
        int ix_sample = ix / units;
        output[ix] = T(0);
        for (int ix_input = 0; ix_input < input_width; ix_input++)
        {
          T mul = clip_fp8_e4m3(inputs[ix_sample*input_width+ix_input]) * clip_fp8_e4m3(weights[ix_input*units+ix_unit]);
          output[ix] = fp32_add(mul, output[ix]);
        }  
    }
};
// fp32 to e5m2 then convert back to fp32
__device__ __forceinline__ float clip_fp8_e5m2(float a) {
    return e5m2_to_fp32(fp32_to_e5m2(a));
}
template <typename T>
__global__ void DenseamKernel_fp8_e5m2(
    const T* inputs,
    const T* weights,
    const int batch, 
    const int units, 
    const int input_width, 
    T* output
    ) 
{ 
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; 
    if(ix < units*batch)
    {
        int ix_unit = ix % units ;
        int ix_sample = ix / units;
        output[ix] = T(0);
        for (int ix_input = 0; ix_input < input_width; ix_input++)
        {
          T mul = clip_fp8_e5m2(inputs[ix_sample*input_width+ix_input]) * clip_fp8_e5m2(weights[ix_input*units+ix_unit]);
          output[ix] = fp32_add(mul, output[ix]);
        }  
    }
};
template <typename T>
__global__ void DenseamWeightsKernel_fp8_e5m2(
    const T* grads,
    const T* inputs,
    const int input_width, 
    const int batch, 
    const int units, 
    T* grad_weights
    ) 
{ 
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; 
    if(ix < units*input_width)
    {
        int ix_unit = ix % units ;
        int ix_input = ix / units;
        grad_weights[ix] = T(0);
        for (int ix_sample = 0; ix_sample < batch; ix_sample++)
        {
            T mul = clip_fp8_e5m2(inputs[input_width*ix_sample+ix_input]) * clip_fp8_e5m2(grads[ix_sample*units+ix_unit]);
            grad_weights[ix] = fp32_add(mul, grad_weights[ix]);
        }  
    }
};
template <typename T>
__global__ void DenseamInputKernel_fp8_e5m2(
    const T* grads,
    const T* weights,
    const int input_width, 
    const int batch, 
    const int units, 
    T* grad_inputs
    ) 
{ 
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; 
    if(ix < batch *input_width)
    {
        int ix_input = ix % input_width;
        int ix_sample = ix / input_width ;
        grad_inputs[ix] = T(0);

        for (int ix_unit = 0; ix_unit < units; ix_unit++)
        {   
            T mul = clip_fp8_e5m2(weights[ix_input*units+ ix_unit]) * clip_fp8_e5m2(grads[ix_sample*units+ix_unit]);
            grad_inputs[ix] = fp32_add(mul, grad_inputs[ix]);
        }
    }
};
/* lut implementation */
/* lut 8-bit exponents typed e.g. Bfloat16*/
#include "AMsimulator.inl"
template <typename T>
__global__ void DenseamKernel_lut(
    const T* inputs,
    const T* weights,
    const int batch, 
    const int units, 
    const int input_width, 
    T* output, 
    cudaTextureObject_t lut,
    const uint32_t mant_mask,
    const uint8_t a_shift,
    const uint8_t b_shift, const uint8_t mant_bitwidth
    ) 
{ 
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; 
    if(ix < units*batch)
    {
        int ix_unit = ix % units ;
        int ix_sample = ix / units;
        output[ix] = T(0);
        for (int ix_input = 0; ix_input < input_width; ix_input++)
        {
            T a = inputs[ix_sample*input_width+ix_input];
            T b = weights[ix_input*units+ix_unit];
            T mul = AMsimulator(a, b, lut, mant_mask, a_shift, b_shift, mant_bitwidth);
            output[ix] = fp32_add(mul, output[ix]);
        }  
    }
};
template <typename T>
__global__ void DenseamWeightsKernel_lut(
    const T* grads,
    const T* inputs,
    const int input_width, 
    const int batch, 
    const int units, 
    T* grad_weights,
    cudaTextureObject_t lut,
    const uint32_t mant_mask,
    const uint8_t a_shift,
    const uint8_t b_shift, const uint8_t mant_bitwidth
    ) 
{ 
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; 
    if(ix < units*input_width)
    {
        int ix_unit = ix % units ;
        int ix_input = ix / units;
        grad_weights[ix] = T(0);
        for (int ix_sample = 0; ix_sample < batch; ix_sample++)
        {
            T a = inputs[input_width*ix_sample+ix_input];
            T b = grads[ix_sample*units+ix_unit];
            T mul = AMsimulator(a, b, lut, mant_mask, a_shift, b_shift, mant_bitwidth);
            grad_weights[ix] = fp32_add(mul, grad_weights[ix]);
        }  
    }
};
template <typename T>
__global__ void DenseamInputKernel_lut(
    const T* grads,
    const T* weights,
    const int input_width, 
    const int batch, 
    const int units, 
    T* grad_inputs, 
    cudaTextureObject_t lut,
    const uint32_t mant_mask,
    const uint8_t a_shift,
    const uint8_t b_shift, const uint8_t mant_bitwidth
    ) 
{ 
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; 
    if(ix < batch *input_width)
    {
        int ix_input = ix % input_width;
        int ix_sample = ix / input_width ;
        grad_inputs[ix] = T(0);

        for (int ix_unit = 0; ix_unit < units; ix_unit++)
        {   
            T a = weights[ix_input*units+ ix_unit];
            T b = grads[ix_sample*units+ix_unit];
            T mul = AMsimulator(a, b, lut, mant_mask, a_shift, b_shift, mant_bitwidth);
            grad_inputs[ix] = fp32_add(mul, grad_inputs[ix]);
        }
    }
};
//lut 5-bit exponents typed e.g. FP16
// a dummy implementation
template <typename T>
__global__ void DenseamKernel_lut_5exp(
    const T* inputs,
    const T* weights,
    const int batch, 
    const int units, 
    const int input_width, 
    T* output, 
    cudaTextureObject_t lut,
    const uint32_t mant_mask,
    const uint8_t a_shift,
    const uint8_t b_shift, const uint8_t mant_bitwidth
    ) 
{ 
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; 
    if(ix < units*batch)
    {
        int ix_unit = ix % units ;
        int ix_sample = ix / units;
        output[ix] = T(0);
        for (int ix_input = 0; ix_input < input_width; ix_input++)
        {
            T a = inputs[ix_sample*input_width+ix_input];
            T b = weights[ix_input*units+ix_unit];
            T mul = clip_fp16(a) * clip_fp16(b);
            output[ix] = fp32_add(mul, output[ix]);
        }  
    }
};
template <typename T>
__global__ void DenseamWeightsKernel_lut_5exp(
    const T* grads,
    const T* inputs,
    const int input_width, 
    const int batch, 
    const int units, 
    T* grad_weights,
    cudaTextureObject_t lut,
    const uint32_t mant_mask,
    const uint8_t a_shift,
    const uint8_t b_shift, const uint8_t mant_bitwidth
    ) 
{ 
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; 
    if(ix < units*input_width)
    {
        int ix_unit = ix % units ;
        int ix_input = ix / units;
        grad_weights[ix] = T(0);
        for (int ix_sample = 0; ix_sample < batch; ix_sample++)
        {
            T a = inputs[input_width*ix_sample+ix_input];
            T b = grads[ix_sample*units+ix_unit];
            T mul = clip_fp16(a) * clip_fp16(b);
            grad_weights[ix] = fp32_add(mul, grad_weights[ix]);
        }  
    }
};
template <typename T>
__global__ void DenseamInputKernel_lut_5exp(
    const T* grads,
    const T* weights,
    const int input_width, 
    const int batch, 
    const int units, 
    T* grad_inputs, 
    cudaTextureObject_t lut,
    const uint32_t mant_mask,
    const uint8_t a_shift,
    const uint8_t b_shift, const uint8_t mant_bitwidth
    ) 
{ 
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; 
    if(ix < batch *input_width)
    {
        int ix_input = ix % input_width;
        int ix_sample = ix / input_width ;
        grad_inputs[ix] = T(0);

        for (int ix_unit = 0; ix_unit < units; ix_unit++)
        {   
            T a = weights[ix_input*units+ ix_unit];
            T b = grads[ix_sample*units+ix_unit];
            T mul = clip_fp16(a) * clip_fp16(b);
            grad_inputs[ix] = fp32_add(mul, grad_inputs[ix]);
        }
    }
};
// gemm lut fp8 kernels
template <typename T>
__global__ void DenseamKernel_lut_e5m2(
    const T* inputs,
    const T* weights,
    const int batch, 
    const int units, 
    const int input_width, 
    T* output, 
    cudaTextureObject_t lut
    ) 
{ 
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; 
    if(ix < units*batch)
    {
        int ix_unit = ix % units ;
        int ix_sample = ix / units;
        output[ix] = T(0);
        for (int ix_input = 0; ix_input < input_width; ix_input++)
        {
            uint8_t a_key = fp32_to_e5m2(inputs[ix_sample*input_width+ix_input]);
            uint8_t b_key = fp32_to_e5m2(weights[ix_input*units+ix_unit]);

            uint32_t index = (a_key << 8) | b_key+256*256;

            float mul_result = tex1Dfetch<float>(lut, index);

            output[ix] += mul_result;
        }  
    }
};
template <typename T>
__global__ void DenseamWeightsKernel_lut_e5m2(
    const T* grads,
    const T* inputs,
    const int input_width, 
    const int batch, 
    const int units, 
    T* grad_weights,
    cudaTextureObject_t lut
    ) 
{ 
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; 
    if(ix < units*input_width)
    {
        int ix_unit = ix % units ;
        int ix_input = ix / units;
        grad_weights[ix] = T(0);
        for (int ix_sample = 0; ix_sample < batch; ix_sample++)
        {
            uint8_t a_key = fp32_to_e5m2(inputs[input_width*ix_sample+ix_input]);
            uint8_t b_key = fp32_to_e5m2(grads[ix_sample*units+ix_unit]);

            uint32_t index = (a_key << 8) | b_key+256*256;

            float mul_result = tex1Dfetch<float>(lut, index);

            grad_weights[ix] += mul_result;
        }  
    }
};
template <typename T>
__global__ void DenseamInputKernel_lut_e5m2(
    const T* grads,
    const T* weights,
    const int input_width, 
    const int batch, 
    const int units, 
    T* grad_inputs, 
    cudaTextureObject_t lut
    ) 
{ 
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; 
    if(ix < batch *input_width)
    {
        int ix_input = ix % input_width;
        int ix_sample = ix / input_width ;
        grad_inputs[ix] = T(0);

        for (int ix_unit = 0; ix_unit < units; ix_unit++)
        {   
            uint8_t a_key = fp32_to_e5m2(weights[ix_input*units+ ix_unit]);
            uint8_t b_key = fp32_to_e5m2(grads[ix_sample*units+ix_unit]);

            uint32_t index = (a_key << 8) | b_key;

            float mul_result = tex1Dfetch<float>(lut, index);

            grad_inputs[ix] += mul_result;
        }
    }
};
template <typename T>
__global__ void DenseamKernel_lut_e4m3(
    const T* inputs,
    const T* weights,
    const int batch, 
    const int units, 
    const int input_width, 
    T* output, 
    cudaTextureObject_t lut
    ) 
{ 
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; 
    if(ix < units*batch)
    {
        int ix_unit = ix % units ;
        int ix_sample = ix / units;
        output[ix] = T(0);
        for (int ix_input = 0; ix_input < input_width; ix_input++)
        {
            uint8_t a_key = fp32_to_e4m3(inputs[ix_sample*input_width+ix_input]);
            uint8_t b_key = fp32_to_e4m3(weights[ix_input*units+ix_unit]);

            uint32_t index = (a_key << 8) | b_key;

            float mul_result = tex1Dfetch<float>(lut, index);

            output[ix] += mul_result;
        }  
    }
};


// Functor for forward pass
template <typename T>
void DenseamFunctor<GpuDevice, T>::operator()(
        const GpuDevice& d, const T* inputs, const T* weights, T* output,
        const int batch, const int units, const int input_width,
        approx_mul_lut<GpuDevice>& mul_lut, FloatMode mode)
{ 
    unsigned blocksize = 1024;
    unsigned gridsize = (batch*units+blocksize -1)/blocksize;
    // //print floatmode
    // std::cout << "FloatMode: " << FloatModeToString(mode) << std::endl;
    // // print if lut is used
    // std::cout << "LUT: " << mul_lut.is_lut() << std::endl;
    // check if mul_lut
    if (mul_lut.is_lut()){
        // using case for different float modes
        switch (mode){
            case FloatMode::FP8E5M2:  
                // use DenseamKernel_lut_e5m2 with lut for both forward pass  
                DenseamKernel_lut_e5m2<T><<<gridsize, blocksize, 0, d.stream()>>>(inputs, weights, batch, units, input_width, output, mul_lut.get_mant_mul_lut_text_());
                break;
            case FloatMode::FP8HYB:
                // use DenseamKernel_lut_e4m3 with lut for forward pass    
                DenseamKernel_lut_e4m3<T><<<gridsize, blocksize, 0, d.stream()>>>(inputs, weights, batch, units, input_width, output, mul_lut.get_mant_mul_lut_text_());
                break;
            case FloatMode::FP16:
                // use denseamkernel_5exp with lut for both forward pass
                DenseamKernel_lut_5exp<T><<<gridsize, blocksize, 0, d.stream()>>>(inputs, weights, batch, units, input_width, output, mul_lut.get_mant_mul_lut_text_(), mul_lut.get_mant_mask_(), mul_lut.get_a_shift_(), mul_lut.get_b_shift_(), mul_lut.get_mant_width_());
                break;
            case FloatMode::BF16:
                // use DenseamKernel_lut with lut for both forward pass
                DenseamKernel_lut<T><<<gridsize, blocksize, 0, d.stream()>>>(inputs, weights, batch, units, input_width, output, mul_lut.get_mant_mul_lut_text_(), mul_lut.get_mant_mask_(), mul_lut.get_a_shift_(), mul_lut.get_b_shift_(), mul_lut.get_mant_width_());
                break;
            case FloatMode::FP32:
                // use DenseamKernel for forward pass
                DenseamKernel<T><<<gridsize, blocksize, 0, d.stream()>>>(inputs, weights, batch, units, input_width, output);
                break;
            default:
                break;

        }
    } else {
        // using case for different float modes
        // no lut all accurate
        switch (mode){
            case FloatMode::FP8E5M2:  
                // use DenseamKernel_fp8_e5m2 without lut for both forward pass  
                DenseamKernel_fp8_e5m2<T><<<gridsize, blocksize, 0, d.stream()>>>(inputs, weights, batch, units, input_width, output);
                break;
            case FloatMode::FP8HYB:
                // use DenseamKernel_fp8_e4m3 without lut for forward pass
                DenseamKernel_fp8_e4m3<T><<<gridsize, blocksize, 0, d.stream()>>>(inputs, weights, batch, units, input_width, output);
                break;
            case FloatMode::FP16:
                // use DenseamKernel_fp16 without lut for both forward pass
                DenseamKernel_fp16<T><<<gridsize, blocksize, 0, d.stream()>>>(inputs, weights, batch, units, input_width, output);
                break;
            case FloatMode::BF16:
                // use DenseamKernel_bf16 without lut for both forward pass
                DenseamKernel_bf16<T><<<gridsize, blocksize, 0, d.stream()>>>(inputs, weights, batch, units, input_width, output);
                break;
            case FloatMode::FP32:
                // use DenseamKernel for forward pass
                DenseamKernel<T><<<gridsize, blocksize, 0, d.stream()>>>(inputs, weights, batch, units, input_width, output);
                break;
            default:
                break;
        }
    }
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

// Functor for weight gradients
template <typename T>
void DenseamWeightGradFunctor<GpuDevice, T>::operator()
    (const GpuDevice& d, const T* inputs, const T* grads,
            T* output, const int batch, const int units, const int input_width,
            approx_mul_lut<GpuDevice>& mul_lut, FloatMode mode )
{
    unsigned blocksize = 1024;
    unsigned gridsize = (units*input_width+blocksize -1)/blocksize;
    // check if mul_lut
    if (mul_lut.is_lut()){
        // using case for different float modes
        switch (mode){
            case FloatMode::FP8E5M2:  
                // use DenseamWeightsKernel_lut_e5m2 with lut for backward pass
                DenseamWeightsKernel_lut_e5m2<T><<<gridsize, blocksize, 0, d.stream()>>>(grads, inputs, input_width, batch, units, output, mul_lut.get_mant_mul_lut_text_());
                break;
            case FloatMode::FP8HYB:
                // use DenseamWeightsKernel_lut_e5m2 with lut for backward pass    
                DenseamWeightsKernel_lut_e5m2<T><<<gridsize, blocksize, 0, d.stream()>>>(grads, inputs, input_width, batch, units, output, mul_lut.get_mant_mul_lut_text_());
                break;
            case FloatMode::FP16:
                // use denseamweightskernel_5exp with lut for backward pass
                DenseamWeightsKernel_lut_5exp<T><<<gridsize, blocksize, 0, d.stream()>>>(grads, inputs, input_width, batch, units, output, mul_lut.get_mant_mul_lut_text_(), mul_lut.get_mant_mask_(), mul_lut.get_a_shift_(), mul_lut.get_b_shift_(), mul_lut.get_mant_width_());
                break;
            case FloatMode::BF16:
                // use DenseamWeightsKernel_lut with lut for backward pass
                DenseamWeightsKernel_lut<T><<<gridsize, blocksize, 0, d.stream()>>>(grads, inputs, input_width, batch, units, output, mul_lut.get_mant_mul_lut_text_(), mul_lut.get_mant_mask_(), mul_lut.get_a_shift_(), mul_lut.get_b_shift_(), mul_lut.get_mant_width_());
                break;
            case FloatMode::FP32:
                // use DenseamWeightsKernel for backward pass
                DenseamWeightsKernel<T><<<gridsize, blocksize, 0, d.stream()>>>(grads, inputs, input_width, batch, units, output);
                break;
            default:
                break;

        }
    } else {
        // using case for different float modes
        // no lut all accurate
        switch(mode){
            case FloatMode::FP8E5M2:  
                // use DenseamWeightsKernel_fp8_e5m2 without lut for backward pass  
                DenseamWeightsKernel_fp8_e5m2<T><<<gridsize, blocksize, 0, d.stream()>>>(grads, inputs, input_width, batch, units, output);
                break;
            case FloatMode::FP8HYB:
                // use DenseamWeightsKernel_fp8_e5m2 without lut for backward pass
                DenseamWeightsKernel_fp8_e5m2<T><<<gridsize, blocksize, 0, d.stream()>>>(grads, inputs, input_width, batch, units, output);
                break;
            case FloatMode::FP16:
                // use DenseamWeightsKernel_fp16 without lut for both backward pass
                DenseamWeightsKernel_fp16<T><<<gridsize, blocksize, 0, d.stream()>>>(grads, inputs, input_width, batch, units, output);
                break;
            case FloatMode::BF16:
                // use DenseamWeightsKernel_bf16 without lut for both backward pass
                DenseamWeightsKernel_bf16<T><<<gridsize, blocksize, 0, d.stream()>>>(grads, inputs, input_width, batch, units, output);
                break;
            case FloatMode::FP32:
                // use DenseamWeightsKernel for backward pass
                DenseamWeightsKernel<T><<<gridsize, blocksize, 0, d.stream()>>>(grads, inputs, input_width, batch, units, output);
                break;
            default:
                break;
        }
    }
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

// Functor for input gradients
template <typename T>
void DenseamInputGradFunctor<GpuDevice, T>::operator()
    (const GpuDevice& d, const T* weights, const T* grads,
            T* output, const int batch, const int units, const int input_width,
            approx_mul_lut<GpuDevice>& mul_lut, FloatMode mode)
{
    unsigned blocksize = 1024;
    unsigned gridsize = (batch*input_width+blocksize -1)/blocksize;

    // check if mul_lut
    if (mul_lut.is_lut()){
        // using case for different float modes
        switch (mode){
            case FloatMode::FP8E5M2:  
                // use DenseamInputKernel_lut_e5m2 with lut for backward pass
                DenseamInputKernel_lut_e5m2<T><<<gridsize, blocksize, 0, d.stream()>>>(grads, weights, input_width, batch, units, output, mul_lut.get_mant_mul_lut_text_());
                break;
            case FloatMode::FP8HYB:
                // use DenseamInputKernel_lut_e5m2 with lut for backward pass    
                DenseamInputKernel_lut_e5m2<T><<<gridsize, blocksize, 0, d.stream()>>>(grads, weights, input_width, batch, units, output, mul_lut.get_mant_mul_lut_text_());
                break;
            case FloatMode::FP16:
                // use denseaminputkernel_5exp with lut for backward pass
                DenseamInputKernel_lut_5exp<T><<<gridsize, blocksize, 0, d.stream()>>>(grads, weights, input_width, batch, units, output, mul_lut.get_mant_mul_lut_text_(), mul_lut.get_mant_mask_(), mul_lut.get_a_shift_(), mul_lut.get_b_shift_(), mul_lut.get_mant_width_());
                break;
            case FloatMode::BF16:
                // use DenseamInputKernel_lut with lut for backward pass
                DenseamInputKernel_lut<T><<<gridsize, blocksize, 0, d.stream()>>>(grads, weights, input_width, batch, units, output, mul_lut.get_mant_mul_lut_text_(), mul_lut.get_mant_mask_(), mul_lut.get_a_shift_(), mul_lut.get_b_shift_(), mul_lut.get_mant_width_());
                break;
            case FloatMode::FP32:
                // use DenseamInputKernel for backward pass
                DenseamInputKernel<T><<<gridsize, blocksize, 0, d.stream()>>>(grads, weights, input_width, batch, units, output);
                break;
            default:
                break;

        }
    } else {
        // using case for different float modes
        // no lut all accurate
        switch(mode){
            case FloatMode::FP8E5M2:  
                // use DenseamInputKernel_fp8_e5m2 without lut for backward pass  
                DenseamInputKernel_fp8_e5m2<T><<<gridsize, blocksize, 0, d.stream()>>>(grads, weights, input_width, batch, units, output);
                break;
            case FloatMode::FP8HYB:
                // use DenseamInputKernel_fp8_e5m2 without lut for backward pass
                DenseamInputKernel_fp8_e5m2<T><<<gridsize, blocksize, 0, d.stream()>>>(grads, weights, input_width, batch, units, output);
                break;
            case FloatMode::FP16:
                // use DenseamInputKernel_fp16 without lut for backward pass
                DenseamInputKernel_fp16<T><<<gridsize, blocksize, 0, d.stream()>>>(grads, weights, input_width, batch, units, output);
                break;
            case FloatMode::BF16:
                // use DenseamInputKernel_bf16 without lut for backward pass
                DenseamInputKernel_bf16<T><<<gridsize, blocksize, 0, d.stream()>>>(grads, weights, input_width, batch, units, output);
                break;
            case FloatMode::FP32:
                // use DenseamInputKernel for backward pass
                DenseamInputKernel<T><<<gridsize, blocksize, 0, d.stream()>>>(grads, weights, input_width, batch, units, output);
                break;
            default:
                break;
        }
    }

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

// Template instantiations
template struct DenseamFunctor<GpuDevice, float>;
template struct DenseamFunctor<GpuDevice, int32>;
template struct DenseamInputGradFunctor<GpuDevice, float>;
template struct DenseamInputGradFunctor<GpuDevice, int32>;
template struct DenseamWeightGradFunctor<GpuDevice, float>;
template struct DenseamWeightGradFunctor<GpuDevice, int32>;
