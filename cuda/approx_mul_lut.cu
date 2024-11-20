#define EIGEN_USE_GPU
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <iterator>
#include "error.cuh"
#include "approx_mul_lut.h"
using namespace tensorflow;

// Constructor Implementation
template<>
approx_mul_lut<Eigen::GpuDevice>::approx_mul_lut(tensorflow::OpKernelConstruction* context)
    : approx_mul_lut_base(context) {
    if (!lut_)
        return;
    if (fp8_) {
        // Handle combined FP8 LUT
        gpuErrchk(cudaMalloc(&mant_mul_lut_cuda_fp32_,
                             mant_mul_lut_fp32_.size() * sizeof(float)));
        gpuErrchk(cudaMemcpy(mant_mul_lut_cuda_fp32_, mant_mul_lut_fp32_.data(),
                             mant_mul_lut_fp32_.size() * sizeof(float),
                             cudaMemcpyHostToDevice));

        // Create texture object for combined FP8 LUT
        cudaResourceDesc res_desc;
        memset(&res_desc, 0, sizeof(cudaResourceDesc));
        res_desc.resType = cudaResourceTypeLinear;
        res_desc.res.linear.devPtr = mant_mul_lut_cuda_fp32_;
        res_desc.res.linear.desc.f = cudaChannelFormatKindFloat;
        res_desc.res.linear.desc.x = 32;
        res_desc.res.linear.sizeInBytes =
            mant_mul_lut_fp32_.size() * sizeof(float);

        cudaTextureDesc tex_desc;
        memset(&tex_desc, 0, sizeof(cudaTextureDesc));
        tex_desc.readMode = cudaReadModeElementType;

        gpuErrchk(cudaCreateTextureObject(&mant_mul_lut_text_,
                                          &res_desc,
                                          &tex_desc, nullptr));
    } else {
        // Handle 8-bit LUT
        gpuErrchk(cudaMalloc(&mant_mul_lut_cuda_uint8_,
                             mant_mul_lut_uint8_.size() * sizeof(uint8_t)));
        gpuErrchk(cudaMemcpy(mant_mul_lut_cuda_uint8_, mant_mul_lut_uint8_.data(),
                             mant_mul_lut_uint8_.size() * sizeof(uint8_t),
                             cudaMemcpyHostToDevice));

        // Create texture object for 8-bit LUT
        cudaResourceDesc res_desc;
        memset(&res_desc, 0, sizeof(cudaResourceDesc));
        res_desc.resType = cudaResourceTypeLinear;
        res_desc.res.linear.devPtr = mant_mul_lut_cuda_uint8_;
        res_desc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
        res_desc.res.linear.desc.x = 8;
        res_desc.res.linear.sizeInBytes =
            mant_mul_lut_uint8_.size() * sizeof(uint8_t);

        cudaTextureDesc tex_desc;
        memset(&tex_desc, 0, sizeof(cudaTextureDesc));
        tex_desc.readMode = cudaReadModeElementType;

        gpuErrchk(cudaCreateTextureObject(&mant_mul_lut_text_,
                                          &res_desc,
                                          &tex_desc, nullptr));
    }
}

// Destructor Implementation
template<>
approx_mul_lut<Eigen::GpuDevice>::~approx_mul_lut() {
    // check if lut is enabled
    if (!lut_)
        return;
    cudaDestroyTextureObject(mant_mul_lut_text_);
    if (fp8_) {
        cudaFree(mant_mul_lut_cuda_fp32_);
    } else {
        cudaFree(mant_mul_lut_cuda_uint8_);
    }
}
