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
    approx_mul_lut<GPUDevice>& mul_lut,
    FloatMode mode,
    bool forward_pass,
    bool input_grad,
    AccumMode accum_mode,
    size_t trunk_size = 0
){

    dim3 blockSize(16, 16, 1);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);
    /* Note: FP8 quantization happens prior to GEMM operations, see cuda_kernel.cu */
    if (mul_lut.is_lut()){
        // using case for different float modes
        switch (mode){
            case FloatMode::FP8E5M2:
            case FloatMode::FP8HYB:
                switch (accum_mode) {
                    case AccumMode::RNE:
                    gemm<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    break;
                    case AccumMode::RZ:
                    gemm_rz<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    break;
                    case AccumMode::FP16RNE:
                    if (trunk_size == TILE_DIM) {
                        gemm_fp16_accumulate_trunksize_16<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    } else if (trunk_size == TRUNK_DIM_32) {
                        dim3 blockSize(TRUNK_DIM_32, TRUNK_DIM_32, 1);
                        dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);
                        gemm_fp16_accumulate_trunksize_32<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    } else if (trunk_size == TRUNK_DIM_64) {
                        gemm_fp16_accumulate_trunksize_64<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    } else {
                        gemm_fp16_accumulate<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    }
                    break;
                    case AccumMode::FP16RZ:
                    if (trunk_size == TILE_DIM) {
                        gemm_fp16_accumulate_rz_trunksize_16<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    } else if (trunk_size == TRUNK_DIM_32) {
                        dim3 blockSize(TRUNK_DIM_32, TRUNK_DIM_32, 1);
                        dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);
                        gemm_fp16_accumulate_rz_trunksize_32<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    } else if (trunk_size == TRUNK_DIM_64) {
                        gemm_fp16_accumulate_rz_trunksize_64<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    } else
                    {
                        gemm_fp16_accumulate_rz<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    }
                    break;
                    case AccumMode::BF16RNE:
                    if (trunk_size == TILE_DIM) {
                        gemm_bf16_accumulate_trunksize_16<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    } else if (trunk_size == TRUNK_DIM_32) {
                        dim3 blockSize(32, 32, 1);
                        dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);
                        gemm_bf16_accumulate_trunksize_32<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    } else if (trunk_size == TRUNK_DIM_64) {
                        gemm_bf16_accumulate_trunksize_64<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    } else if (trunk_size != 0) {
                        gemm_bf16_accumulate_trunksize_x<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc, trunk_size);
                    }
                    else {
                        gemm_bf16_accumulate<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    }
                    break;
                    case AccumMode::BF16RZ:
                    if (trunk_size == TILE_DIM) {
                        gemm_bf16_accumulate_rz_trunksize_16<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    } else if (trunk_size == TRUNK_DIM_32) {
                        dim3 blockSize(TRUNK_DIM_32, TRUNK_DIM_32, 1);
                        dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);
                        gemm_bf16_accumulate_rz_trunksize_32<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    } else if (trunk_size == TRUNK_DIM_64) {
                        gemm_bf16_accumulate_rz_trunksize_64<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    } else if (trunk_size != 0) {
                        gemm_bf16_accumulate_rz_trunksize_x<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc, trunk_size);
                    } else{
                        gemm_bf16_accumulate_rz<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    }
                    break;
                    
                    case AccumMode::SEAFP16RZ:
                        if (trunk_size != 0) {
                            sea_gemm_fp16_accumulate_rz_trunksize_x<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc, trunk_size);
                        } else {
                            sea_gemm_fp16_accumulate_rz<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                        }
                    break;
                    case AccumMode::SEABF16RNE:
                        if (trunk_size != 0) {
                            sea_gemm_bf16_accumulate_trunksize_x<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc, trunk_size);
                        } else {
                            sea_gemm_bf16_accmulate<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                        }
                    break;

                    case AccumMode::SEABF16RZ:
                        if (trunk_size != 0) {
                            sea_gemm_bf16_accumulate_rz_trunksize_x<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc, trunk_size);
                        } else {
                            sea_gemm_bf16_accumulate_rz<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                        }
                    break;

                    default:
                    gemm<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    break;
                }
                break;
            case FloatMode::FP16:
                // use gemm_5exp with lut for both forward and backward pass
                // TODO: add actual implementation
                //exit
                
                switch (accum_mode)
                {
                    case AccumMode::RNE:
                    gemm_5exp<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc, mul_lut.get_mant_mul_lut_text_(), mul_lut.get_mant_mask_(), mul_lut.get_a_shift_(), mul_lut.get_b_shift_(), mul_lut.get_mant_width_());
                    break;
                    case AccumMode::RZ:
                    gemm_5exp<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc, mul_lut.get_mant_mul_lut_text_(), mul_lut.get_mant_mask_(), mul_lut.get_a_shift_(), mul_lut.get_b_shift_(), mul_lut.get_mant_width_());
                    break;
                    default:
                    gemm_5exp<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc, mul_lut.get_mant_mul_lut_text_(), mul_lut.get_mant_mask_(), mul_lut.get_a_shift_(), mul_lut.get_b_shift_(), mul_lut.get_mant_width_());
                    break;
                }
                break;
            case FloatMode::BF16:
                // use gemm with lut for both forward and backward pass
                // the gemm supports 8-bit exponents, mantissa from 0 to 7bits, where 7bit is the bfloat16
                switch (accum_mode)
                {
                        case AccumMode::RNE:
                        gemm<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc, mul_lut.get_mant_mul_lut_text_(), mul_lut.get_mant_mask_(), mul_lut.get_a_shift_(), mul_lut.get_b_shift_(), mul_lut.get_mant_width_());
                        break;
                        case AccumMode::RZ:
                        gemm_rz<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc, mul_lut.get_mant_mul_lut_text_(), mul_lut.get_mant_mask_(), mul_lut.get_a_shift_(), mul_lut.get_b_shift_(), mul_lut.get_mant_width_());
                        break;
                        default:
                        gemm<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc, mul_lut.get_mant_mul_lut_text_(), mul_lut.get_mant_mask_(), mul_lut.get_a_shift_(), mul_lut.get_b_shift_(), mul_lut.get_mant_width_());
                        break;
                }
                break;
            case FloatMode::FP32:
                // use gemm no lut is supported, if you really want to use some approximation, a behavior model is required that is C/C++ based and can be used in the kernel
                switch (accum_mode)
                {
                        case AccumMode::SEARNE:
                        if (trunk_size != 0) {
                            sea_gemm_accumulate_trunksize_x<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc, trunk_size);
                        } else {
                            gemm<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                        }
                        break;
                        case AccumMode::RNE:
                        if (trunk_size == TILE_DIM) {
                            gemm_accumulate_trunksize_16<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                        } else if (trunk_size == TRUNK_DIM_32) {
                            dim3 blockSize(TRUNK_DIM_32, TRUNK_DIM_32, 1);
                            dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);
                            gemm_accumulate_trunksize_32<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                        } else if (trunk_size == TRUNK_DIM_64) {
                            gemm_accumulate_trunksize_64<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                        } else {
                            gemm<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                        }
                        break;
                        case AccumMode::RZ:
                        gemm_rz<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                        break;
                        default:
                        gemm<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                        break;
                }
                break;
            default:
                break;
        }
    } else {
        switch (mode){
            case FloatMode::FP8E5M2:
            case FloatMode::FP8HYB:
                switch (accum_mode) {
                    case AccumMode::RNE:
                    gemm<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    break;
                    case AccumMode::RZ:
                    gemm_rz<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    break;
                    case AccumMode::FP16RNE:
                    if (trunk_size == TILE_DIM) {
                        gemm_fp16_accumulate_trunksize_16<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    } else if (trunk_size == TRUNK_DIM_32) {
                        dim3 blockSize(TRUNK_DIM_32, TRUNK_DIM_32, 1);
                        dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);
                        gemm_fp16_accumulate_trunksize_32<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    } else if (trunk_size == TRUNK_DIM_64) {
                        gemm_fp16_accumulate_trunksize_64<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    } else {
                        gemm_fp16_accumulate<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    }
                    break;
                    case AccumMode::FP16RZ:
                    if (trunk_size == TILE_DIM) {
                        gemm_fp16_accumulate_rz_trunksize_16<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    } else if (trunk_size == TRUNK_DIM_32) {
                        dim3 blockSize(TRUNK_DIM_32, TRUNK_DIM_32, 1);
                        dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);
                        gemm_fp16_accumulate_rz_trunksize_32<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    } else if (trunk_size == TRUNK_DIM_64) {
                        gemm_fp16_accumulate_rz_trunksize_64<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    } else if (trunk_size != 0) {
                        gemm_fp16_accumulate_rz_trunksize_x<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc, trunk_size);
                    } else {
                        gemm_fp16_accumulate_rz<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    }
                    break;
                    case AccumMode::BF16RNE:
                    if (trunk_size == TILE_DIM) {
                        gemm_bf16_accumulate_trunksize_16<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    } else if (trunk_size == TRUNK_DIM_32) {
                        dim3 blockSize(32, 32, 1);
                        dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);
                        gemm_bf16_accumulate_trunksize_32<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    } else if (trunk_size == TRUNK_DIM_64) {
                        gemm_bf16_accumulate_trunksize_64<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    } else if (trunk_size != 0) {
                        gemm_bf16_accumulate_trunksize_x<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc, trunk_size);
                    }
                    else {
                        gemm_bf16_accumulate<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    }
                    break;
                    case AccumMode::BF16RZ:
                    if (trunk_size == TILE_DIM) {
                        gemm_bf16_accumulate_rz_trunksize_16<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    } else if (trunk_size == TRUNK_DIM_32) {
                        dim3 blockSize(TRUNK_DIM_32, TRUNK_DIM_32, 1);
                        dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);
                        gemm_bf16_accumulate_rz_trunksize_32<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    } else if (trunk_size == TRUNK_DIM_64) {
                        gemm_bf16_accumulate_rz_trunksize_64<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    } else if (trunk_size != 0) {
                        gemm_bf16_accumulate_rz_trunksize_x<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc, trunk_size);
                    } else{
                        gemm_bf16_accumulate_rz<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    }
                    break;
                    case AccumMode::SEAFP16RZ:
                        if (trunk_size != 0) {
                            sea_gemm_fp16_accumulate_rz_trunksize_x<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc, trunk_size);
                        } else {
                            sea_gemm_fp16_accumulate_rz<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                        }
                    break;
                    case AccumMode::SEABF16RNE:
                        if (trunk_size != 0) {
                            sea_gemm_bf16_accumulate_trunksize_x<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc, trunk_size);
                        } else {
                            sea_gemm_bf16_accmulate<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                        }
                    break;

                    case AccumMode::SEABF16RZ:
                        if (trunk_size != 0) {
                            sea_gemm_bf16_accumulate_rz_trunksize_x<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc, trunk_size);
                        } else {
                            sea_gemm_bf16_accumulate_rz<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                        }
                    break;
                    default:
                    gemm<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    break;
                }
                break;
            case FloatMode::FP16:
                switch (accum_mode)
                {
                    case AccumMode::RNE:
                    gemm_fp16<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    break;
                    case AccumMode::RZ:
                    gemm_fp16_rz<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    break;
                    default:
                    gemm_fp16<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    break;
                }
                break;
            case FloatMode::BF16:
                switch (accum_mode)
                {
                    case AccumMode::RNE:
                    gemm_bf16<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    break;
                    case AccumMode::RZ:
                    gemm_bf16_rz<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    break;
                    default:
                    gemm_bf16<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    break;
                }
                break;
            case FloatMode::FP32:
                switch (accum_mode)
                {
                    case AccumMode::SEARNE:
                        if (trunk_size != 0) {
                            sea_gemm_accumulate_trunksize_x<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc, trunk_size);
                        } else {
                            gemm<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                        }
                        break;
                    case AccumMode::RNE:
                        if (trunk_size == TILE_DIM) {
                            gemm_accumulate_trunksize_16<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                        } else if (trunk_size == TRUNK_DIM_32) {
                            dim3 blockSize(TRUNK_DIM_32, TRUNK_DIM_32, 1);
                            dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);
                            gemm_accumulate_trunksize_32<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                        } else if (trunk_size == TRUNK_DIM_64) {
                            gemm_accumulate_trunksize_64<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                        } else if (trunk_size != 0) {
                            gemm_accumulate_trunksize_x<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc, trunk_size);
                        } else {
                            gemm<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                        }
                    break;
                    case AccumMode::RZ:
                    gemm_rz<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    break;
                    default:
                    gemm<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, a, lda, b, ldb, c, ldc);
                    break;
                }
                break;
            default:
                break;
        }

    }
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

#endif  // GEMM_LAUNCHER_CUH