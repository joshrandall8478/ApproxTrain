

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
// start of new kernels


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
// non-lut fp32_rz
template <typename T>
__global__ void DenseamKernel_rz(
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
          output[ix] = fp32_add_rz(mul, output[ix]);
        }  
    }
};
template <typename T>
__global__ void DenseamWeightsKernel_rz(
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
            grad_weights[ix] = fp32_add_rz(mul, grad_weights[ix]);
        }  
    }
};
template <typename T>
__global__ void DenseamInputKernel_rz(
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
            grad_inputs[ix] = fp32_add_rz(mul, grad_inputs[ix]);
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
// non-lut bf16 rz
template <typename T>
__global__ void DenseamKernel_bf16_rz(
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
          output[ix] = fp32_add_rz(mul, output[ix]);
        }  
    }
};
template <typename T>
__global__ void DenseamWeightsKernel_bf16_rz(
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
            grad_weights[ix] = fp32_add_rz(mul, grad_weights[ix]);
        }  
    }
};
template <typename T>
__global__ void DenseamInputKernel_bf16_rz(
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
            grad_inputs[ix] = fp32_add_rz(mul, grad_inputs[ix]);
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
// non-lut fp16_rz
template <typename T>
__global__ void DenseamKernel_fp16_rz(
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
          output[ix] = fp32_add_rz(mul, output[ix]);
        }  
    }
};
template <typename T>
__global__ void DenseamWeightsKernel_fp16_rz(
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
            grad_weights[ix] = fp32_add_rz(mul, grad_weights[ix]);
        }  
    }
};
template <typename T>
__global__ void DenseamInputKernel_fp16_rz(
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
            grad_inputs[ix] = fp32_add_rz(mul, grad_inputs[ix]);
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
/* lut rz implementation */
/* lut 8-bit exponents typed e.g. Bfloat16 rz*/
template <typename T>
__global__ void DenseamKernel_lut_rz(
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
            output[ix] = fp32_add_rz(mul, output[ix]);
        }  
    }
};
template <typename T>
__global__ void DenseamWeightsKernel_lut_rz(
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
            grad_weights[ix] = fp32_add_rz(mul, grad_weights[ix]);
        }  
    }
};
template <typename T>
__global__ void DenseamInputKernel_lut_rz(
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
            grad_inputs[ix] = fp32_add_rz(mul, grad_inputs[ix]);
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
/* these kernels are intended for pairing with lower precision (fp8)*/
// bf16 accumulate 
template <typename T>
__global__ void DenseamKernel_bf16_accumulate(
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
          output[ix] = bf16_add(mul, output[ix]);
        }  
    }
};
template <typename T>
__global__ void DenseamWeightsKernel_bf16_accumulate(
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
            grad_weights[ix] = bf16_add(mul, grad_weights[ix]);
        }  
    }
};
template <typename T>
__global__ void DenseamInputKernel_bf16_accumulate(
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
            grad_inputs[ix] = bf16_add(mul, grad_inputs[ix]);
        }
    }
};
// bf16 accumulate rz
template <typename T>
__global__ void DenseamKernel_bf16_accumulate_rz(
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
          output[ix] = bf16_add_rz(mul, output[ix]);
        }  
    }
};
template <typename T>
__global__ void DenseamWeightsKernel_bf16_accumulate_rz(
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
            grad_weights[ix] = bf16_add_rz(mul, grad_weights[ix]);
        }  
    }
};
template <typename T>
__global__ void DenseamInputKernel_bf16_accumulate_rz(
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
            grad_inputs[ix] = bf16_add_rz(mul, grad_inputs[ix]);
        }
    }
};
// fp16 accumulate
template <typename T>
__global__ void DenseamKernel_fp16_accumulate(
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
          T mul = inputs[ix_sample*input_width+ix_input]* weights[ix_input*units+ix_unit];
          output[ix] = half_add(mul, output[ix]);
        }  
    }
};
template <typename T>
__global__ void DenseamWeightsKernel_fp16_accumulate(
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
            grad_weights[ix] = half_add(mul, grad_weights[ix]);
        }  
    }
};
template <typename T>
__global__ void DenseamInputKernel_fp16_accumulate(
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
            grad_inputs[ix] = half_add(mul, grad_inputs[ix]);
        }
    }
};
// fp16 accumulate rz
template <typename T>
__global__ void DenseamKernel_fp16_accumulate_rz(
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
          T mul = inputs[ix_sample*input_width+ix_input]* weights[ix_input*units+ix_unit];
          output[ix] = half_add_rz(mul, output[ix]);
        }  
    }
};
template <typename T>
__global__ void DenseamWeightsKernel_fp16_accumulate_rz(
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
            grad_weights[ix] = half_add_rz(mul, grad_weights[ix]);
        }  
    }
};
template <typename T>
__global__ void DenseamInputKernel_fp16_accumulate_rz(
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
            grad_inputs[ix] = half_add_rz(mul, grad_inputs[ix]);
        }
    }
};
// Functor for forward pass
template <typename T>
void DenseamFunctor<GpuDevice, T>::operator()(
        const GpuDevice& d, const T* inputs, const T* weights, T* output,
        const int batch, const int units, const int input_width,
        approx_mul_lut<GpuDevice>& mul_lut, FloatMode mode, T* quant_input, T* quant_weight, AccumMode accum_mode
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
                quant_fp32_e5m2clipping_launcher<T>(d, inputs, quant_input, input_size);
                quant_fp32_e5m2clipping_launcher<T>(d, weights, quant_weight, weight_size);
                break;
            case FloatMode::FP8HYB:
                quant_fp32_e4m3clipping_launcher<T>(d, inputs, quant_input, input_size);
                quant_fp32_e4m3clipping_launcher<T>(d, weights, quant_weight, weight_size);
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
    GEMM_LAUNCHER<T>(d, m, n, k, input_data, lda, weight_data, ldb, output, ldc, blockSize, gridSize, mul_lut, mode, false, false, accum_mode);
}

// Functor for weight gradients
template <typename T>
void DenseamWeightGradFunctor<GpuDevice, T>::operator()
    (const GpuDevice& d, const T* inputs, const T* grads,
            T* output, const int batch, const int units, const int input_width,
            approx_mul_lut<GpuDevice>& mul_lut, FloatMode mode, T* quant_input, T* quant_grad, AccumMode accum_mode, T *input_T)
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
                quant_fp32_e5m2clipping_launcher<T>(d, input_T, quant_input, input_size);
                quant_fp32_e5m2clipping_launcher<T>(d, grads, quant_grad, grad_size);
                break;
            case FloatMode::FP8HYB:
                quant_fp32_e4m3clipping_launcher<T>(d, input_T, quant_input, input_size);
                quant_fp32_e4m3clipping_launcher<T>(d, grads, quant_grad, grad_size);
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
    GEMM_LAUNCHER<T>(d, m, n, k, input_data, lda, grad_data, ldb, output, ldc, blockSize, gridSize, mul_lut, mode, true, false, accum_mode);
    
}

// Functor for input gradients
template <typename T>
void DenseamInputGradFunctor<GpuDevice, T>::operator()
    (const GpuDevice& d, const T* weights, const T* grads,
            T* output, const int batch, const int units, const int input_width,
            approx_mul_lut<GpuDevice>& mul_lut, FloatMode mode, T* quant_weight, T* quant_grad, AccumMode accum_mode, T* weight_T)
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
                quant_fp32_e5m2clipping_launcher<T>(d, weight_T, quant_weight, weight_size);
                quant_fp32_e5m2clipping_launcher<T>(d, grads, quant_grad, grad_size);
                break;
            case FloatMode::FP8HYB:
                quant_fp32_e4m3clipping_launcher<T>(d, weight_T, quant_weight, weight_size);
                quant_fp32_e4m3clipping_launcher<T>(d, grads, quant_grad, grad_size);
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
    GEMM_LAUNCHER<T>(d, m, n, k, grad_data, lda, weight_data, ldb, output, ldc, blockSize, gridSize, mul_lut, mode, false, true, accum_mode);
}

// Template instantiations
template struct DenseamFunctor<GpuDevice, float>;
template struct DenseamFunctor<GpuDevice, int32>;
template struct DenseamInputGradFunctor<GpuDevice, float>;
template struct DenseamInputGradFunctor<GpuDevice, int32>;
template struct DenseamWeightGradFunctor<GpuDevice, float>;
template struct DenseamWeightGradFunctor<GpuDevice, int32>;
