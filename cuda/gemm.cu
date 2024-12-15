#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "tensorflow/core/framework/types.h"
#include <cuda_fp16.h>
#include "fp8_conversion.cuh"
using namespace tensorflow;




#define TILE_DIM 16
/*
    Goal is to setup emulation with (FP16,FP32), (BF16,FP32), (E4M3, FP32), (E5M2, FP32),
    where the FP32 is accumulation type and others are the input types (multiplications type).
*/

#ifdef RTZ
    #define fp32_add(a,b) __fadd_rz((a), (b));
#else
    #define fp32_add(a,b) ((a)+(b));
#endif
__device__ __forceinline__ float bf16_add_rz(float val, float b) {
    float ret = __fadd_rz(val, b);
    return __uint_as_float(__float_as_uint(ret) & 0xffff0000);
}

__device__ __forceinline__ float half_add_rz(float val, float b) {
    float ret = __fadd_rz(val, b);
    return __half2float(__float2half_rz(ret));
}

/*
    The following is for half precision that accumulate in FP32
*/

/* start of new implementation*/
/* gemm non-lut fp32 */
template <typename T>
__global__ void gemm(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc
   )
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
            // Accumulate the result

            T mul = As[threadIdx.y][n]*Bs[n][threadIdx.x];
            value = fp32_add(value, mul);

         }
    
         __syncthreads();
    }
    
    if (Row < m && Col < n) {
         c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
    }
}
// fp32 initialisation of gemm
template __global__ void gemm<float>(size_t m, size_t n, size_t k,
   const float *a, size_t lda, const float *b, size_t ldb,
   float *c, size_t ldc
   );
// int32 initialisation of gemm
template __global__ void gemm<int32>(size_t m, size_t n, size_t k,
   const int32 *a, size_t lda, const int32 *b, size_t ldb,
   int32 *c, size_t ldc
   );

/* gemm non-lut bfloat16*/
// clip the values to bf16 then convert it back to fp32 (truncation)
__device__ __forceinline__ float clip_bf16(float a) {
    return __uint_as_float(__float_as_uint(a) & 0xffff0000);
}

template <typename T>
__global__ void gemm_bf16(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc
   )
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
            T mul = clip_bf16(As[threadIdx.y][n])*clip_bf16(Bs[n][threadIdx.x]);
            value = fp32_add(value, mul);
         }
    
         __syncthreads();
    }
    
    if (Row < m && Col < n) {
         c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
    }
}
// bf16 initialisation of gemm
template __global__ void gemm_bf16<float>(size_t m, size_t n, size_t k,
   const float *a, size_t lda, const float *b, size_t ldb,
   float *c, size_t ldc
   );
// int32 initialisation of gemm
template __global__ void gemm_bf16<int32>(size_t m, size_t n, size_t k,
   const int32 *a, size_t lda, const int32 *b, size_t ldb,
   int32 *c, size_t ldc
   );


/* gemm non-lut fp16*/
// clip the values to fp16 then convert it back to fp32 (truncation)
__device__ __forceinline__ float clip_fp16(float a) {
    return __half2float(__float2half(a));
}

template <typename T>
__global__ void gemm_fp16(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc
   )
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
            T mul = clip_fp16(As[threadIdx.y][n])*clip_fp16(Bs[n][threadIdx.x]);
            value = fp32_add(value, mul);
         }
    
         __syncthreads();
    }
    
    if (Row < m && Col < n) {
         c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
    }
}
// fp16 initialisation of gemm
template __global__ void gemm_fp16<float>(size_t m, size_t n, size_t k,
   const float *a, size_t lda, const float *b, size_t ldb,
   float *c, size_t ldc
   );
// int32 initialisation of gemm
template __global__ void gemm_fp16<int32>(size_t m, size_t n, size_t k,
   const int32 *a, size_t lda, const int32 *b, size_t ldb,
   int32 *c, size_t ldc
   );


/* lut implementation */
/* gemm lut 8-bit exponents typed e.g. Bfloat16*/
#include "AMsimulator.inl"
template <typename T>
__global__ void gemm(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc, cudaTextureObject_t mant_lut,
   uint32_t mant_mask, uint8_t a_shift, uint8_t b_shift, uint8_t mant_bitwidth
   ){
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
            // use am simulator
            T mul = AMsimulator(As[threadIdx.y][n], Bs[n][threadIdx.x], mant_lut, mant_mask, a_shift, b_shift, mant_bitwidth);
            // Accumulate the result
            value = fp32_add(value, mul);
         }
    
         __syncthreads();
    }
    
    if (Row < m && Col < n) {
         c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
    }
}
// lut initialisation of gemm
template __global__ void gemm<float>(size_t m, size_t n, size_t k,
   const float *a, size_t lda, const float *b, size_t ldb,
   float *c, size_t ldc, cudaTextureObject_t mant_lut,
   uint32_t mant_mask, uint8_t a_shift, uint8_t b_shift, uint8_t mant_bitwidth
   );
// int32 initialisation of gemm
template __global__ void gemm<int32>(size_t m, size_t n, size_t k,
   const int32 *a, size_t lda, const int32 *b, size_t ldb,
   int32 *c, size_t ldc, cudaTextureObject_t mant_lut,
   uint32_t mant_mask, uint8_t a_shift, uint8_t b_shift, uint8_t mant_bitwidth
   );

// gemm lut 5-bit exponents typed e.g. FP16
// this is a dummy implementation
template <typename T>
__global__ void gemm_5exp(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc, cudaTextureObject_t mant_lut,
   uint32_t mant_mask, uint8_t a_shift, uint8_t b_shift, uint8_t mant_bitwidth
   ){
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
            // dummy implementation
            T mul = clip_fp16(As[threadIdx.y][n])*clip_fp16(Bs[n][threadIdx.x]);
            value = fp32_add(value, mul);   
         }
    
         __syncthreads();
    }
    
    if (Row < m && Col < n) {
         c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
    }
}
// initialisation of gemm_5exp
template __global__ void gemm_5exp<float>(size_t m, size_t n, size_t k,
   const float *a, size_t lda, const float *b, size_t ldb,
   float *c, size_t ldc, cudaTextureObject_t mant_lut,
   uint32_t mant_mask, uint8_t a_shift, uint8_t b_shift, uint8_t mant_bitwidth
   );
// int32 initialisation of gemm_5exp
template __global__ void gemm_5exp<int32>(size_t m, size_t n, size_t k,
   const int32 *a, size_t lda, const int32 *b, size_t ldb,
   int32 *c, size_t ldc, cudaTextureObject_t mant_lut,
   uint32_t mant_mask, uint8_t a_shift, uint8_t b_shift, uint8_t mant_bitwidth
   );

/* end of new implementation*/