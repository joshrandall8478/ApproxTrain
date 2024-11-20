

#define EIGEN_USE_GPU

#include "gpu_kernel_helper.h"
#include "error.cuh"
#include "denseam.h"
#include "approx_mul_lut.h"
#include <cuda_fp16.h>
#include "fp8_conversion.cuh"
using namespace tensorflow;
using GpuDevice = Eigen::GpuDevice;

#ifdef AMSIMULATOR
   #define MULTIPLY(a,b) AMsimulator((a), (b), lut, mant_mask, a_shift, b_shift, mant_bitwidth);
   #include "AMsimulator.inl"
#else
   #define MULTIPLY(a,b) ((a)*(b));
#endif

#ifdef RTZ
    #define fp32_add(a,b) __fadd_rz((a), (b));
#else
    #define fp32_add(a,b) ((a)+(b));
#endif

// Convert FP32 to FP16
__device__ __half fp32_to_fp16(float a) {
    return __float2half(a);
}

// Convert FP16 back to FP32
__device__ float fp16_to_fp32(__half a) {
    return __half2float(a);
}

// Forward kernel for e5m2
template <typename T>
__global__ void Denseam_e5m2_Kernel(
    const T* inputs,
    const T* weights,
    const int batch, 
    const int units, 
    const int input_width, 
    T* output, 
    cudaTextureObject_t lut) 
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

// Backward kernel for weights, e5m2
template <typename T>
__global__ void DenseamWeights_e5m2_Kernel(
    const T* grads,
    const T* inputs,
    const int input_width, 
    const int batch, 
    const int units, 
    T* grad_weights,
    cudaTextureObject_t lut) 
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

// Backward kernel for inputs, e5m2
template <typename T>
__global__ void DenseamInput_e5m2_Kernel(
    const T* grads,
    const T* weights,
    const int input_width, 
    const int batch, 
    const int units, 
    T* grad_inputs, 
    cudaTextureObject_t lut) 
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

            uint32_t index = (a_key << 8) | b_key+256*256;

            float mul_result = tex1Dfetch<float>(lut, index);

            grad_inputs[ix] += mul_result;
        }
    }
};

// Original kernel (non-FP8)
template <typename T>
__global__ void DenseamKernel(
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
          #ifdef FP16MUL
                __half a_fp16 = fp32_to_fp16(inputs[ix_sample*input_width+ix_input]);
                __half b_fp16 = fp32_to_fp16(weights[ix_input*units+ix_unit]);
                float mul_fp32 = fp16_to_fp32(__hmul(a_fp16, b_fp16));

                output[ix] = fp32_add(mul_fp32, output[ix]);
          #else
            T mul = MULTIPLY(inputs[ix_sample*input_width+ix_input], weights[ix_input*units+ix_unit]);
            output[ix] = fp32_add(mul, output[ix]);
          #endif
        }  
    }
};

// Original weight gradient kernel (non-FP8)
template <typename T>
__global__ void DenseamWeightsKernel(
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
            #ifdef FP16MUL
                __half a_fp16 = fp32_to_fp16(inputs[input_width*ix_sample+ix_input]);
                __half b_fp16 = fp32_to_fp16(grads[ix_sample*units+ix_unit]);
                float mul_fp32 = fp16_to_fp32(__hmul(a_fp16, b_fp16));

                grad_weights[ix] = fp32_add(mul_fp32, grad_weights[ix]);
            #else
                T mul = MULTIPLY(inputs[input_width*ix_sample+ix_input], grads[ix_sample*units+ix_unit]);
                grad_weights[ix] = fp32_add(mul, grad_weights[ix]);
            #endif
        }  
    }
};

// Original input gradient kernel (non-FP8)
template <typename T>
__global__ void DenseamInputKernel(
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
            #ifdef FP16MUL
                __half a_fp16 = fp32_to_fp16(weights[ix_input*units+ ix_unit]);
                __half b_fp16 = fp32_to_fp16(grads[ix_sample*units+ix_unit]);
                float mul_fp32 = fp16_to_fp32(__hmul(a_fp16, b_fp16));

                grad_inputs[ix] = fp32_add(mul_fp32, grad_inputs[ix]);
            #else
                T mul = MULTIPLY(weights[ix_input*units+ ix_unit], grads[ix_sample*units+ix_unit]);
                grad_inputs[ix] = fp32_add(mul, grad_inputs[ix]);
            #endif 
        }
    }
};

// Functor for forward pass
template <typename T>
void DenseamFunctor<GpuDevice, T>::operator()(
        const GpuDevice& d, const T* inputs, const T* weights, T* output,
        const int batch, const int units, const int input_width,
        approx_mul_lut<GpuDevice>& mul_lut, bool fp8)
{ 
    unsigned blocksize = 1024;
    unsigned gridsize = (batch*units+blocksize -1)/blocksize;

    if (fp8)
    {
            Denseam_e5m2_Kernel<T><<<gridsize, blocksize, 0, d.stream()>>>(inputs, weights, batch, units, input_width, output, mul_lut.get_mant_mul_lut_text_());
    }
    else
    {
        const uint32_t mant_mask = mul_lut.get_mant_mask_();
        const uint8_t a_shift = mul_lut.get_a_shift_();
        const uint8_t b_shift = mul_lut.get_b_shift_();
        const uint8_t mant_bitwidth = mul_lut.get_mant_width_();
        DenseamKernel<T><<<gridsize, blocksize, 0, d.stream()>>>(inputs, weights, batch, units, input_width, output, mul_lut.get_mant_mul_lut_text_(), mant_mask, a_shift, b_shift, mant_bitwidth);
    }
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

// Functor for weight gradients
template <typename T>
void DenseamWeightGradFunctor<GpuDevice, T>::operator()
    (const GpuDevice& d, const T* inputs, const T* grads,
            T* output, const int batch, const int units, const int input_width,
            approx_mul_lut<GpuDevice>& mul_lut, bool fp8) 
{
    unsigned blocksize = 1024;
    unsigned gridsize = (units*input_width+blocksize -1)/blocksize;

    if (fp8)
    {
        DenseamWeights_e5m2_Kernel<T><<<gridsize, blocksize, 0, d.stream()>>>(grads, inputs, input_width, batch, units, output, mul_lut.get_mant_mul_lut_text_());
    }
    else
    {
        const uint32_t mant_mask = mul_lut.get_mant_mask_();
        const uint8_t a_shift = mul_lut.get_a_shift_();
        const uint8_t b_shift = mul_lut.get_b_shift_();
        const uint8_t mant_bitwidth = mul_lut.get_mant_width_();
        DenseamWeightsKernel<T><<<gridsize, blocksize, 0, d.stream()>>>(grads, inputs, input_width, batch, units, output, mul_lut.get_mant_mul_lut_text_(), mant_mask, a_shift, b_shift, mant_bitwidth);
    }
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

// Functor for input gradients
template <typename T>
void DenseamInputGradFunctor<GpuDevice, T>::operator()
    (const GpuDevice& d, const T* weights, const T* grads,
            T* output, const int batch, const int units, const int input_width,
            approx_mul_lut<GpuDevice>& mul_lut, bool fp8)
{
    unsigned blocksize = 1024;
    unsigned gridsize = (batch*input_width+blocksize -1)/blocksize;

    if (fp8)
    {
        DenseamInput_e5m2_Kernel<T><<<gridsize, blocksize, 0, d.stream()>>>(grads, weights, input_width, batch, units, output, mul_lut.get_mant_mul_lut_text_());
    }
    else
    {
        const uint32_t mant_mask = mul_lut.get_mant_mask_();
        const uint8_t a_shift = mul_lut.get_a_shift_();
        const uint8_t b_shift = mul_lut.get_b_shift_();
        const uint8_t mant_bitwidth = mul_lut.get_mant_width_();
        DenseamInputKernel<T><<<gridsize, blocksize, 0, d.stream()>>>(grads, weights, input_width, batch, units, output, mul_lut.get_mant_mul_lut_text_(), mant_mask, a_shift, b_shift, mant_bitwidth);
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
