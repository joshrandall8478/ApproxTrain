#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cuda_fp16.h>
#include "tensorflow/core/framework/types.h"
using namespace tensorflow;

#ifdef AMSIMULATOR
   #define MULTIPLY(a,b) AMsimulator((a), (b), mant_lut, mant_mask, a_shift, b_shift, mant_bitwidth);
   #include "AMsimulator.inl"
#else
   #define MULTIPLY(a,b) ((a)*(b));
#endif


#define TILE_DIM 16
/*
    Goal is to setup SEA emulation with (FP16,FP32), (BF16,FP32), (E4M3, FP32), (E5M2, FP32)
    Poritise (BF16, FP32), then somehow we generalise implementation
*/

#ifdef RTZ
    #define fp32_add(a,b) __fadd_rz((a), (b));
#else
    #define fp32_add(a,b) ((a)+(b));
#endif




/*
    The following is for half precision that accumulate in FP32
*/


// Convert FP32 to FP16
__device__ __half fp32_to_fp16(float a) {
    return __float2half(a);
}

// Convert FP16 back to FP32
__device__ float fp16_to_fp32(__half a) {
    return __half2float(a);
}


            

template <typename T>
__global__ void gemm(size_t m, size_t n, size_t k,
    const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc, cudaTextureObject_t mant_lut,
   uint32_t mant_mask, uint8_t a_shift, uint8_t b_shift, uint8_t mant_bitwidth)
{
    T value(0);

    int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    int Col = blockIdx.x*TILE_DIM + threadIdx.x;

    __shared__ T As[TILE_DIM][TILE_DIM];
    __shared__ T Bs[TILE_DIM][TILE_DIM];

    for (int i = 0; i < (TILE_DIM + k - 1)/TILE_DIM; ++i) {

        if (i*TILE_DIM + threadIdx.x < k && Row < m){
            As[threadIdx.y][threadIdx.x] = a[Row*lda + i*TILE_DIM + threadIdx.x];
        }
        else{
            As[threadIdx.y][threadIdx.x] = T(0);
        }

        if (i*TILE_DIM + threadIdx.y < k && Col < n){
            Bs[threadIdx.y][threadIdx.x] = b[(i*TILE_DIM + threadIdx.y)*ldb + Col];
        }
        else{
            Bs[threadIdx.y][threadIdx.x] = T(0);
        }

        __syncthreads();

        for (int n = 0; n < TILE_DIM; ++n){
            #ifdef FP16MUL
                // Convert to FP16, perform multiplication, convert back to FP32
                __half a_fp16 = fp32_to_fp16(As[threadIdx.y][n]);
                __half b_fp16 = fp32_to_fp16(Bs[n][threadIdx.x]);
                float mul_fp32 = fp16_to_fp32(__hmul(a_fp16, b_fp16));

                // Accumulate in FP32
                value = fp32_add(mul_fp32, value);
            #else
                T mul = MULTIPLY(As[threadIdx.y][n], Bs[n][threadIdx.x]);
                value = fp32_add(mul, value);
            #endif
        }

        __syncthreads();
    }

    if (Row < m && Col < n) {
        c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
    }
}

template __global__ void gemm<float>(size_t m, size_t n, size_t k,
    const float *a, size_t lda, const float *b, size_t ldb,
   float *c, size_t ldc, cudaTextureObject_t mant_lut,
   uint32_t mant_mask, uint8_t a_shift, uint8_t b_shift, uint8_t mant_bitwidth);
template __global__ void gemm<int32>(size_t m, size_t n, size_t k,
    const int32 *a, size_t lda, const int32 *b, size_t ldb,
   int32 *c, size_t ldc, cudaTextureObject_t mant_lut,
   uint32_t mant_mask, uint8_t a_shift, uint8_t b_shift, uint8_t mant_bitwidth);
