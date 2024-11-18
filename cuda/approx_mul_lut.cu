#define EIGEN_USE_GPU
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <iterator>
#include "error.cuh"
#include "approx_mul_lut.h"
using namespace tensorflow;

template<>
class approx_mul_lut<Eigen::GpuDevice> : public approx_mul_lut_base {
    public:
        explicit approx_mul_lut(tensorflow::OpKernelConstruction* context);
        ~approx_mul_lut();
        auto get_mant_mul_lut_() -> void* {
            if (fp8_) {
                return mant_mul_lut_cuda_fp32_;
            } else {
                return mant_mul_lut_cuda_uint8_;
            }
        }

        auto get_mant_mul_lut_text_() -> cudaTextureObject_t& {
            return mant_mul_lut_text_;
        }
};


approx_mul_lut<Eigen::GpuDevice>::approx_mul_lut(OpKernelConstruction* context)
    : approx_mul_lut_base(context) {
    if (fp8_) {
        // Handle 32-bit float LUT
        // Allocate CUDA memory for the FP32 LUT
        gpuErrchk(cudaMalloc(&mant_mul_lut_cuda_fp32_,
                             mant_mul_lut_fp32_.size() * sizeof(float)));
        gpuErrchk(cudaMemcpy(mant_mul_lut_cuda_fp32_, mant_mul_lut_fp32_.data(),
                             mant_mul_lut_fp32_.size() * sizeof(float),
                             cudaMemcpyHostToDevice));

        // Create texture object for FP32 LUT
        cudaResourceDesc mant_mul_lut_res_desc;
        memset(&mant_mul_lut_res_desc, 0, sizeof(cudaResourceDesc));
        mant_mul_lut_res_desc.resType = cudaResourceTypeLinear;
        mant_mul_lut_res_desc.res.linear.devPtr = mant_mul_lut_cuda_fp32_;
        mant_mul_lut_res_desc.res.linear.desc.f = cudaChannelFormatKindFloat;
        mant_mul_lut_res_desc.res.linear.desc.x = 32;
        mant_mul_lut_res_desc.res.linear.sizeInBytes =
            mant_mul_lut_fp32_.size() * sizeof(float);

        cudaTextureDesc mant_mul_text_desc;
        memset(&mant_mul_text_desc, 0, sizeof(cudaTextureDesc));
        mant_mul_text_desc.readMode = cudaReadModeElementType;

        gpuErrchk(cudaCreateTextureObject(&mant_mul_lut_text_,
                                          &mant_mul_lut_res_desc,
                                          &mant_mul_text_desc, nullptr));
    } else {
        // Handle 8-bit LUT
        // Allocate CUDA memory for the uint8 LUT
        gpuErrchk(cudaMalloc(&mant_mul_lut_cuda_uint8_,
                             mant_mul_lut_uint8_.size() * sizeof(uint8_t)));
        gpuErrchk(cudaMemcpy(mant_mul_lut_cuda_uint8_, mant_mul_lut_uint8_.data(),
                             mant_mul_lut_uint8_.size() * sizeof(uint8_t),
                             cudaMemcpyHostToDevice));

        // Create texture object for uint8 LUT
        cudaResourceDesc mant_mul_lut_res_desc;
        memset(&mant_mul_lut_res_desc, 0, sizeof(cudaResourceDesc));
        mant_mul_lut_res_desc.resType = cudaResourceTypeLinear;
        mant_mul_lut_res_desc.res.linear.devPtr = mant_mul_lut_cuda_uint8_;
        mant_mul_lut_res_desc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
        mant_mul_lut_res_desc.res.linear.desc.x = 8;
        mant_mul_lut_res_desc.res.linear.sizeInBytes =
            mant_mul_lut_uint8_.size() * sizeof(uint8_t);

        cudaTextureDesc mant_mul_text_desc;
        memset(&mant_mul_text_desc, 0, sizeof(cudaTextureDesc));
        mant_mul_text_desc.readMode = cudaReadModeElementType;

        gpuErrchk(cudaCreateTextureObject(&mant_mul_lut_text_,
                                          &mant_mul_lut_res_desc,
                                          &mant_mul_text_desc, nullptr));
    }
}


approx_mul_lut<Eigen::GpuDevice>::~approx_mul_lut() {
    cudaDestroyTextureObject(mant_mul_lut_text_);
    if (fp8_) {
        cudaFree(mant_mul_lut_cuda_fp32_);
    } else {
        cudaFree(mant_mul_lut_cuda_uint8_);
    }
}