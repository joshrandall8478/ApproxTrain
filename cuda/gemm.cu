#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "tensorflow/core/framework/types.h"
#include <cuda_fp16.h>
#include "fp8_conversion.cuh"
#include "accumulate.cuh"
#include "gemm.cuh"
using namespace tensorflow;





/* start of new implementation*/
/* gemm bf16 accumulate */
template <typename T>
__global__ void gemm_bf16_accumulate(size_t m, size_t n, size_t k,
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
               value = bf16_add(value, As[threadIdx.y][n]*Bs[n][threadIdx.x]);
         }
    
         __syncthreads();
    }
    
    if (Row < m && Col < n) {
         c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
    }
}
template __global__ void gemm_bf16_accumulate<float>(size_t m, size_t n, size_t k,
   const float *a, size_t lda, const float *b, size_t ldb,
   float *c, size_t ldc
   );
template __global__ void gemm_bf16_accumulate<int32>(size_t m, size_t n, size_t k,
     const int32 *a, size_t lda, const int32 *b, size_t ldb,
     int32 *c, size_t ldc
);
/* gemm bf16 accumulate rtz */
template <typename T>
__global__ void gemm_bf16_accumulate_rz(size_t m, size_t n, size_t k,
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
               value = bf16_add_rz(value, As[threadIdx.y][n]*Bs[n][threadIdx.x]);
         }
    
         __syncthreads();
    }
    
    if (Row < m && Col < n) {
         c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
    }
}
template __global__ void gemm_bf16_accumulate_rz<float>(size_t m, size_t n, size_t k,
   const float *a, size_t lda, const float *b, size_t ldb,
   float *c, size_t ldc
   );
template __global__ void gemm_bf16_accumulate_rz<int32>(size_t m, size_t n, size_t k,
     const int32 *a, size_t lda, const int32 *b, size_t ldb,
     int32 *c, size_t ldc
     );

template <typename T>
__global__ void gemm_fp16_accumulate(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc
   )
{
    float value(0);
    
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
            value = half_add(value, As[threadIdx.y][n]*Bs[n][threadIdx.x]);
         }
    
         __syncthreads();
    }
    
    if (Row < m && Col < n) {
         c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
    }
}
template __global__ void gemm_fp16_accumulate<float>(size_t m, size_t n, size_t k,
   const float *a, size_t lda, const float *b, size_t ldb,
   float *c, size_t ldc
   );
template __global__ void gemm_fp16_accumulate<int32>(size_t m, size_t n, size_t k,
     const int32 *a, size_t lda, const int32 *b, size_t ldb,
     int *c, size_t ldc
     );
// #define TRUNK_DIM 32
// template <typename T>
// __global__ void gemm_fp16_accumulate_rz(
//     size_t m, size_t n, size_t k,
//     const T * a, size_t lda,
//     const T * b, size_t ldb,
//     T * c, size_t ldc
// )
// {
//     // Thread's output coordinates
//     int Row = blockIdx.y * TRUNK_DIM + threadIdx.y;
//     int Col = blockIdx.x * TRUNK_DIM + threadIdx.x;

//     // Accumulator for the final output element
//     T value = T(0);

//     // Shared memory tiles for A and B
//     __shared__ T As[TRUNK_DIM][TRUNK_DIM];
//     __shared__ T Bs[TRUNK_DIM][TRUNK_DIM];

//     // Number of tiles along the K dimension
//     int numTiles = (k + TRUNK_DIM - 1) / TRUNK_DIM;

//     for (int tileIdx = 0; tileIdx < numTiles; ++tileIdx)
//     {
//         // Load tile from A
//         int A_col = tileIdx * TRUNK_DIM + threadIdx.x;
//         if ((Row < m) && (A_col < k)) {
//             As[threadIdx.y][threadIdx.x] = a[Row * lda + A_col];
//         } else {
//             As[threadIdx.y][threadIdx.x] = T(0);
//         }

//         // Load tile from B
//         int B_row = tileIdx * TRUNK_DIM + threadIdx.y;
//         if ((Col < n) && (B_row < k)) {
//             Bs[threadIdx.y][threadIdx.x] = b[B_row * ldb + Col];
//         } else {
//             Bs[threadIdx.y][threadIdx.x] = T(0);
//         }

//         __syncthreads();

//         // Local tile accumulator for this iteration
//         T tileSum = T(0);

//         // Sum partial products for the current tile
//         #pragma unroll
//         for (int idx = 0; idx < TRUNK_DIM; ++idx) {
//             tileSum += As[threadIdx.y][idx] * Bs[idx][threadIdx.x];
//         }

//         // Add this tile's contribution to the final value
//         value += tileSum;

//         __syncthreads();
//     }

//     // Write final result (Row, Col) if within matrix bounds
//     if (Row < m && Col < n) {
//         c[Row * ldc + Col] = value;
//     }
// }

/* gemm fp16 accumulate */
template <typename T>
__global__ void gemm_fp16_accumulate_rz(size_t m, size_t n, size_t k,
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
          // trunk accumulation (TILE_DIM)
          // tileSum = half_add_rz(tileSum, As[threadIdx.y][n]*Bs[n][threadIdx.x]);
          value = half_add_rz(value, As[threadIdx.y][n]*Bs[n][threadIdx.x]);


         }
          // value = half_add_rz(tileSum, value);

         __syncthreads();
    }

    if (Row < m && Col < n) {
         c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
    }
}
template __global__ void gemm_fp16_accumulate_rz<float>(size_t m, size_t n, size_t k,
   const float *a, size_t lda, const float *b, size_t ldb,
   float *c, size_t ldc
   );
template __global__ void gemm_fp16_accumulate_rz<int32>(size_t m, size_t n, size_t k,
     const int32 *a, size_t lda, const int32 *b, size_t ldb,
     int32 *c, size_t ldc
     );
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

template <typename T>
__global__ void gemm_rz(size_t m, size_t n, size_t k,
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
            value = fp32_add_rz(value, mul);
         }
    
         __syncthreads();
    }
    
    if (Row < m && Col < n) {
         c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
    }
}
template __global__ void gemm_rz<float>(size_t m, size_t n, size_t k,
   const float *a, size_t lda, const float *b, size_t ldb,
   float *c, size_t ldc
   );
template __global__ void gemm_rz<int32>(size_t m, size_t n, size_t k,
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

template <typename T>
__global__ void gemm_bf16_rz(size_t m, size_t n, size_t k,
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
            value = fp32_add_rz(value, mul);
         }
    
         __syncthreads();
    }
    
    if (Row < m && Col < n) {
         c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
    }
}

template  __global__ void gemm_bf16_rz<float>(size_t m, size_t n, size_t k,
   const float *a, size_t lda, const float *b, size_t ldb,
   float *c, size_t ldc
   );
template  __global__ void gemm_bf16_rz<int32>(size_t m, size_t n, size_t k,
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

// implement gemm_fp16_rz
template <typename T>
__global__ void gemm_fp16_rz(size_t m, size_t n, size_t k,
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
            value = fp32_add_rz(value, mul);
         }
    
         __syncthreads();
    }
    
    if (Row < m && Col < n) {
         c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
    }
}
template __global__ void gemm_fp16_rz<float>(size_t m, size_t n, size_t k,
   const float *a, size_t lda, const float *b, size_t ldb,
   float *c, size_t ldc
   );
template __global__ void gemm_fp16_rz<int32>(size_t m, size_t n, size_t k,
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
// lut rz
template <typename T>
__global__ void gemm_rz(size_t m, size_t n, size_t k,
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
            value = fp32_add_rz(value, mul);
         }
    
         __syncthreads();
    }
    
    if (Row < m && Col < n) {
         c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
    }
}
template __global__ void gemm_rz<float>(size_t m, size_t n, size_t k,
   const float *a, size_t lda, const float *b, size_t ldb,
   float *c, size_t ldc, cudaTextureObject_t mant_lut,
   uint32_t mant_mask, uint8_t a_shift, uint8_t b_shift, uint8_t mant_bitwidth
   );
template __global__ void gemm_rz<int32>(size_t m, size_t n, size_t k,
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
/* below kernels are intended for trunk based accumulation*/
/* intended for using withing hybfp8 quantization kernels */
template <typename T> 
__global__ void gemm_fp16_accumulate_rz_trunksize_16(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc) {
     T value(0);
     int Row = blockIdx.y * TILE_DIM + threadIdx.y;
     int Col = blockIdx.x * TILE_DIM + threadIdx.x;
     __shared__ T As[TILE_DIM][TILE_DIM];
     __shared__ T Bs[TILE_DIM][TILE_DIM];
     for (int i = 0; i < (TILE_DIM + k - 1) / TILE_DIM; ++i) {
          if (i * TILE_DIM + threadIdx.x < k && Row < m) {
               As[threadIdx.y][threadIdx.x] = a[Row * lda + i * TILE_DIM + threadIdx.x];
          } else {
               As[threadIdx.y][threadIdx.x] = T(0);
          }
          if (i * TILE_DIM + threadIdx.y < k && Col < n) {
               Bs[threadIdx.y][threadIdx.x] = b[(i * TILE_DIM + threadIdx.y) * ldb + Col];
          } else {
               Bs[threadIdx.y][threadIdx.x] = T(0);
          }
          __syncthreads();
          T tileSum = T(0);
          for (int n = 0; n < TILE_DIM; ++n) {
               T mul = As[threadIdx.y][n]*Bs[n][threadIdx.x];
               tileSum = half_add_rz(tileSum, mul);
          }
          value = half_add_rz(tileSum, value);
          __syncthreads();
     }
     if (Row < m && Col < n) {
          c[((blockIdx.y * blockDim.y + threadIdx.y) * ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
     }
}
template __global__ void gemm_fp16_accumulate_rz_trunksize_16<float>(size_t m, size_t n, size_t k,
   const float *a, size_t lda, const float *b, size_t ldb,
   float *c, size_t ldc);
template __global__ void gemm_fp16_accumulate_rz_trunksize_16<int32>(size_t m, size_t n, size_t k,
     const int32 *a, size_t lda, const int32 *b, size_t ldb,
     int32 *c, size_t ldc);


template <typename T>
__global__ void gemm_fp16_accumulate_rz_trunksize_32(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc){
     T value(0);
     int Row = blockIdx.y * TRUNK_DIM_32 + threadIdx.y;
     int Col = blockIdx.x * TRUNK_DIM_32 + threadIdx.x;
     __shared__ T As[TRUNK_DIM_32][TRUNK_DIM_32];
     __shared__ T Bs[TRUNK_DIM_32][TRUNK_DIM_32];
     for (int i = 0; i < (TRUNK_DIM_32 + k - 1) / TRUNK_DIM_32; ++i) {
          if (i * TRUNK_DIM_32 + threadIdx.x < k && Row < m) {
               As[threadIdx.y][threadIdx.x] = a[Row * lda + i * TRUNK_DIM_32 + threadIdx.x];
          } else {
               As[threadIdx.y][threadIdx.x] = T(0);
          }
          if (i * TRUNK_DIM_32 + threadIdx.y < k && Col < n) {
               Bs[threadIdx.y][threadIdx.x] = b[(i * TRUNK_DIM_32 + threadIdx.y) * ldb + Col];
          } else {
               Bs[threadIdx.y][threadIdx.x] = T(0);
          }
          __syncthreads();
          T tileSum = T(0);
          for (int n = 0; n < TRUNK_DIM_32; ++n) {
               T mul = As[threadIdx.y][n]*Bs[n][threadIdx.x];
               tileSum = half_add_rz(tileSum, mul);
          }
          value = half_add_rz(tileSum, value);
          __syncthreads();
     }
     if (Row < m && Col < n) {
          c[((blockIdx.y * blockDim.y + threadIdx.y) * ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
     }
   }
template __global__ void gemm_fp16_accumulate_rz_trunksize_32<float>(size_t m, size_t n, size_t k,
     const float *a, size_t lda, const float *b, size_t ldb,
     float *c, size_t ldc);
template __global__ void gemm_fp16_accumulate_rz_trunksize_32<int32>(size_t m, size_t n, size_t k,
     const int32 *a, size_t lda, const int32 *b, size_t ldb,
     int32 *c, size_t ldc);

template <typename T>
__global__ void gemm_bf16_accumulate_rz_trunksize_16(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc){
     T value(0);
     int Row = blockIdx.y * TILE_DIM + threadIdx.y;
     int Col = blockIdx.x * TILE_DIM + threadIdx.x;
     __shared__ T As[TILE_DIM][TILE_DIM];
     __shared__ T Bs[TILE_DIM][TILE_DIM];
     for (int i = 0; i < (TILE_DIM + k - 1) / TILE_DIM; ++i) {
          if (i * TILE_DIM + threadIdx.x < k && Row < m) {
               As[threadIdx.y][threadIdx.x] = a[Row * lda + i * TILE_DIM + threadIdx.x];
          } else {
               As[threadIdx.y][threadIdx.x] = T(0);
          }
          if (i * TILE_DIM + threadIdx.y < k && Col < n) {
               Bs[threadIdx.y][threadIdx.x] = b[(i * TILE_DIM + threadIdx.y) * ldb + Col];
          } else {
               Bs[threadIdx.y][threadIdx.x] = T(0);
          }
          __syncthreads();
          T tileSum = T(0);
          for (int n = 0; n < TILE_DIM; ++n) {
               T mul = As[threadIdx.y][n]*Bs[n][threadIdx.x];
               tileSum = bf16_add_rz(tileSum, mul);
          }
          value = bf16_add_rz(tileSum, value);
          __syncthreads();
     }
     if (Row < m && Col < n) {
          c[((blockIdx.y * blockDim.y + threadIdx.y) * ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
     }
}
template __global__ void gemm_bf16_accumulate_rz_trunksize_16<float>(size_t m, size_t n, size_t k,
     const float *a, size_t lda, const float *b, size_t ldb,
     float *c, size_t ldc);
template __global__ void gemm_bf16_accumulate_rz_trunksize_16<int32>(size_t m, size_t n, size_t k,
     const int32 *a, size_t lda, const int32 *b, size_t ldb,
     int32 *c, size_t ldc);

template <typename T>
__global__ void gemm_bf16_accumulate_rz_trunksize_32(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc){
     T value(0);
     int Row = blockIdx.y * TRUNK_DIM_32 + threadIdx.y;
     int Col = blockIdx.x * TRUNK_DIM_32 + threadIdx.x;
     __shared__ T As[TRUNK_DIM_32][TRUNK_DIM_32];
     __shared__ T Bs[TRUNK_DIM_32][TRUNK_DIM_32];
     for (int i = 0; i < (TRUNK_DIM_32 + k - 1) / TRUNK_DIM_32; ++i) {
          if (i * TRUNK_DIM_32 + threadIdx.x < k && Row < m) {
               As[threadIdx.y][threadIdx.x] = a[Row * lda + i * TRUNK_DIM_32 + threadIdx.x];
          } else {
               As[threadIdx.y][threadIdx.x] = T(0);
          }
          if (i * TRUNK_DIM_32 + threadIdx.y < k && Col < n) {
               Bs[threadIdx.y][threadIdx.x] = b[(i * TRUNK_DIM_32 + threadIdx.y) * ldb + Col];
          } else {
               Bs[threadIdx.y][threadIdx.x] = T(0);
          }
          __syncthreads();
          T tileSum = T(0);
          for (int n = 0; n < TRUNK_DIM_32; ++n) {
               T mul = As[threadIdx.y][n]*Bs[n][threadIdx.x];
               tileSum = bf16_add_rz(tileSum, mul);
          }
          value = bf16_add_rz(tileSum, value);
          __syncthreads();
     }
     if (Row < m && Col < n) {
          c[((blockIdx.y * blockDim.y + threadIdx.y) * ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
     }
   }
template __global__ void gemm_bf16_accumulate_rz_trunksize_32<float>(size_t m, size_t n, size_t k,
     const float *a, size_t lda, const float *b, size_t ldb,
     float *c, size_t ldc);
template __global__ void gemm_bf16_accumulate_rz_trunksize_32<int32>(size_t m, size_t n, size_t k,
     const int32 *a, size_t lda, const int32 *b, size_t ldb,
     int32 *c, size_t ldc);

template <typename T>
__global__ void gemm_bf16_accumulate_trunksize_16(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc){
     T value(0);
     int Row = blockIdx.y * TILE_DIM + threadIdx.y;
     int Col = blockIdx.x * TILE_DIM + threadIdx.x;
     __shared__ T As[TILE_DIM][TILE_DIM];
     __shared__ T Bs[TILE_DIM][TILE_DIM];
     for (int i = 0; i < (TILE_DIM + k - 1) / TILE_DIM; ++i) {
          if (i * TILE_DIM + threadIdx.x < k && Row < m) {
               As[threadIdx.y][threadIdx.x] = a[Row * lda + i * TILE_DIM + threadIdx.x];
          } else {
               As[threadIdx.y][threadIdx.x] = T(0);
          }
          if (i * TILE_DIM + threadIdx.y < k && Col < n) {
               Bs[threadIdx.y][threadIdx.x] = b[(i * TILE_DIM + threadIdx.y) * ldb + Col];
          } else {
               Bs[threadIdx.y][threadIdx.x] = T(0);
          }
          __syncthreads();
          T tileSum = T(0);
          for (int n = 0; n < TILE_DIM; ++n) {
               T mul = As[threadIdx.y][n]*Bs[n][threadIdx.x];
               tileSum = bf16_add(tileSum, mul);
          }
          value = bf16_add(tileSum, value);
          __syncthreads();
     }
     if (Row < m && Col < n) {
          c[((blockIdx.y * blockDim.y + threadIdx.y) * ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
     }
}
template __global__ void gemm_bf16_accumulate_trunksize_16<float>(size_t m, size_t n, size_t k,
     const float *a, size_t lda, const float *b, size_t ldb,
     float *c, size_t ldc);
template __global__ void gemm_bf16_accumulate_trunksize_16<int32>(size_t m, size_t n, size_t k,
     const int32 *a, size_t lda, const int32 *b, size_t ldb,
     int32 *c, size_t ldc);

template <typename T>
__global__ void gemm_bf16_accumulate_trunksize_32(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc){
     T value(0);
     int Row = blockIdx.y * TRUNK_DIM_32 + threadIdx.y;
     int Col = blockIdx.x * TRUNK_DIM_32 + threadIdx.x;
     __shared__ T As[TRUNK_DIM_32][TRUNK_DIM_32];
     __shared__ T Bs[TRUNK_DIM_32][TRUNK_DIM_32];
     for (int i = 0; i < (TRUNK_DIM_32 + k - 1) / TRUNK_DIM_32; ++i) {
          if (i * TRUNK_DIM_32 + threadIdx.x < k && Row < m) {
               As[threadIdx.y][threadIdx.x] = a[Row * lda + i * TRUNK_DIM_32 + threadIdx.x];
          } else {
               As[threadIdx.y][threadIdx.x] = T(0);
          }
          if (i * TRUNK_DIM_32 + threadIdx.y < k && Col < n) {
               Bs[threadIdx.y][threadIdx.x] = b[(i * TRUNK_DIM_32 + threadIdx.y) * ldb + Col];
          } else {
               Bs[threadIdx.y][threadIdx.x] = T(0);
          }
          __syncthreads();
          T tileSum = T(0);
          for (int n = 0; n < TRUNK_DIM_32; ++n) {
               T mul = As[threadIdx.y][n]*Bs[n][threadIdx.x];
               tileSum = bf16_add(tileSum, mul);
          }
          value = bf16_add(tileSum, value);
          __syncthreads();
     }
     if (Row < m && Col < n) {
          c[((blockIdx.y * blockDim.y + threadIdx.y) * ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
     }
}
template __global__ void gemm_bf16_accumulate_trunksize_32<float>(size_t m, size_t n, size_t k,
     const float *a, size_t lda, const float *b, size_t ldb,
     float *c, size_t ldc);
template __global__ void gemm_bf16_accumulate_trunksize_32<int32>(size_t m, size_t n, size_t k,
     const int32 *a, size_t lda, const int32 *b, size_t ldb,
     int32 *c, size_t ldc);

template <typename T>
__global__ void gemm_fp16_accumulate_trunksize_16(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc) {
     T value(0);
     int Row = blockIdx.y * TILE_DIM + threadIdx.y;
     int Col = blockIdx.x * TILE_DIM + threadIdx.x;
     __shared__ T As[TILE_DIM][TILE_DIM];
     __shared__ T Bs[TILE_DIM][TILE_DIM];
     for (int i = 0; i < (TILE_DIM + k - 1) / TILE_DIM; ++i) {
          if (i * TILE_DIM + threadIdx.x < k && Row < m) {
               As[threadIdx.y][threadIdx.x] = a[Row * lda + i * TILE_DIM + threadIdx.x];
          } else {
               As[threadIdx.y][threadIdx.x] = T(0);
          }
          if (i * TILE_DIM + threadIdx.y < k && Col < n) {
               Bs[threadIdx.y][threadIdx.x] = b[(i * TILE_DIM + threadIdx.y) * ldb + Col];
          } else {
               Bs[threadIdx.y][threadIdx.x] = T(0);
          }
          __syncthreads();
          T tileSum = T(0);
          for (int n = 0; n < TILE_DIM; ++n) {
               T mul = As[threadIdx.y][n]*Bs[n][threadIdx.x];
               tileSum = half_add(tileSum, mul);
          }
          value = half_add(tileSum, value);
          __syncthreads();
     }
     if (Row < m && Col < n) {
          c[((blockIdx.y * blockDim.y + threadIdx.y) * ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
     }
}
template __global__ void gemm_fp16_accumulate_trunksize_16<float>(size_t m, size_t n, size_t k,
     const float *a, size_t lda, const float *b, size_t ldb,
     float *c, size_t ldc);
template __global__ void gemm_fp16_accumulate_trunksize_16<int32>(size_t m, size_t n, size_t k,
     const int32 *a, size_t lda, const int32 *b, size_t ldb,
     int32 *c, size_t ldc);


template <typename T>
__global__ void gemm_fp16_accumulate_trunksize_32(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc) {
     T value(0);
     int Row = blockIdx.y * TRUNK_DIM_32 + threadIdx.y;
     int Col = blockIdx.x * TRUNK_DIM_32 + threadIdx.x;
     __shared__ T As[TRUNK_DIM_32][TRUNK_DIM_32];
     __shared__ T Bs[TRUNK_DIM_32][TRUNK_DIM_32];
     for (int i = 0; i < (TRUNK_DIM_32 + k - 1) / TRUNK_DIM_32; ++i) {
          if (i * TRUNK_DIM_32 + threadIdx.x < k && Row < m) {
               As[threadIdx.y][threadIdx.x] = a[Row * lda + i * TRUNK_DIM_32 + threadIdx.x];
          } else {
               As[threadIdx.y][threadIdx.x] = T(0);
          }
          if (i * TRUNK_DIM_32 + threadIdx.y < k && Col < n) {
               Bs[threadIdx.y][threadIdx.x] = b[(i * TRUNK_DIM_32 + threadIdx.y) * ldb + Col];
          } else {
               Bs[threadIdx.y][threadIdx.x] = T(0);
          }
          __syncthreads();
          T tileSum = T(0);
          for (int n = 0; n < TRUNK_DIM_32; ++n) {
               T mul = As[threadIdx.y][n]*Bs[n][threadIdx.x];
               tileSum = half_add(tileSum, mul);
          }
          value = half_add(tileSum, value);
          __syncthreads();
     }
     if (Row < m && Col < n) {
          c[((blockIdx.y * blockDim.y + threadIdx.y) * ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
     }
}
template __global__ void gemm_fp16_accumulate_trunksize_32<float>(size_t m, size_t n, size_t k,
     const float *a, size_t lda, const float *b, size_t ldb,
     float *c, size_t ldc);
template __global__ void gemm_fp16_accumulate_trunksize_32<int32>(size_t m, size_t n, size_t k,
     const int32 *a, size_t lda, const int32 *b, size_t ldb,
     int32 *c, size_t ldc);

template <typename T>
__global__ void gemm_fp16_accumulate_trunksize_64(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc) {
     T value(0);
     int Row = blockIdx.y * TILE_DIM + threadIdx.y;
     int Col = blockIdx.x * TILE_DIM + threadIdx.x;
     __shared__ T As[TILE_DIM][TILE_DIM];
     __shared__ T Bs[TILE_DIM][TILE_DIM];
     int accumulate_step = 64/TILE_DIM;
     T local_accumulator(0);
     for (int i = 0; i < (TILE_DIM + k - 1) / TILE_DIM; ++i) {
          if (i * TILE_DIM + threadIdx.x < k && Row < m) {
               As[threadIdx.y][threadIdx.x] = a[Row * lda + i * TILE_DIM + threadIdx.x];
          } else {
               As[threadIdx.y][threadIdx.x] = T(0);
          }
          if (i * TILE_DIM + threadIdx.y < k && Col < n) {
               Bs[threadIdx.y][threadIdx.x] = b[(i * TILE_DIM + threadIdx.y) * ldb + Col];
          } else {
               Bs[threadIdx.y][threadIdx.x] = T(0);
          }
          __syncthreads();

          for (int n = 0; n < TILE_DIM; ++n) {
               T mul = As[threadIdx.y][n]*Bs[n][threadIdx.x];
               local_accumulator = half_add(local_accumulator, mul);
          }
          if (i % accumulate_step == 0) {
               value = half_add(local_accumulator, value);
               local_accumulator = T(0);
          }
          __syncthreads();
     }
     // if remaining elements are less than 64
     if (local_accumulator != T(0)) {
          value = half_add(local_accumulator, value);
     }
     if (Row < m && Col < n) {
          c[((blockIdx.y * blockDim.y + threadIdx.y) * ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
     }
}
template __global__ void gemm_fp16_accumulate_trunksize_64<float>(size_t m, size_t n, size_t k,
     const float *a, size_t lda, const float *b, size_t ldb,
     float *c, size_t ldc);
template __global__ void gemm_fp16_accumulate_trunksize_64<int32>(size_t m, size_t n, size_t k,
     const int32 *a, size_t lda, const int32 *b, size_t ldb,
     int32 *c, size_t ldc);

template <typename T>
__global__ void gemm_fp16_accumulate_rz_trunksize_64(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc) {
     T value(0);
     int Row = blockIdx.y * TILE_DIM + threadIdx.y;
     int Col = blockIdx.x * TILE_DIM + threadIdx.x;
     __shared__ T As[TILE_DIM][TILE_DIM];
     __shared__ T Bs[TILE_DIM][TILE_DIM];
     int accumulate_step = 64/TILE_DIM;
     T local_accumulator(0);
     for (int i = 0; i < (TILE_DIM + k - 1) / TILE_DIM; ++i) {
          if (i * TILE_DIM + threadIdx.x < k && Row < m) {
               As[threadIdx.y][threadIdx.x] = a[Row * lda + i * TILE_DIM + threadIdx.x];
          } else {
               As[threadIdx.y][threadIdx.x] = T(0);
          }
          if (i * TILE_DIM + threadIdx.y < k && Col < n) {
               Bs[threadIdx.y][threadIdx.x] = b[(i * TILE_DIM + threadIdx.y) * ldb + Col];
          } else {
               Bs[threadIdx.y][threadIdx.x] = T(0);
          }
          __syncthreads();

          for (int n = 0; n < TILE_DIM; ++n) {
               T mul = As[threadIdx.y][n]*Bs[n][threadIdx.x];
               local_accumulator = half_add_rz(local_accumulator, mul);
          }
          if (i % accumulate_step == 0) {
               value = half_add_rz(local_accumulator, value);
               local_accumulator = T(0);
          }
          __syncthreads();
     }
     // if remaining elements are less than 64
     if (local_accumulator != T(0)) {
          value = half_add_rz(local_accumulator, value);
     }
     if (Row < m && Col < n) {
          c[((blockIdx.y * blockDim.y + threadIdx.y) * ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
     }
}
template __global__ void gemm_fp16_accumulate_rz_trunksize_64<float>(size_t m, size_t n, size_t k,
     const float *a, size_t lda, const float *b, size_t ldb,
     float *c, size_t ldc);
template __global__ void gemm_fp16_accumulate_rz_trunksize_64<int32>(size_t m, size_t n, size_t k,
     const int32 *a, size_t lda, const int32 *b, size_t ldb,
     int32 *c, size_t ldc);

template <typename T>
__global__ void gemm_bf16_accumulate_rz_trunksize_64(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc) {
     T value(0);
     int Row = blockIdx.y * TILE_DIM + threadIdx.y;
     int Col = blockIdx.x * TILE_DIM + threadIdx.x;
     __shared__ T As[TILE_DIM][TILE_DIM];
     __shared__ T Bs[TILE_DIM][TILE_DIM];
     int accumulate_step = 64/TILE_DIM;
     T local_accumulator(0);
     for (int i = 0; i < (TILE_DIM + k - 1) / TILE_DIM; ++i) {
          if (i * TILE_DIM + threadIdx.x < k && Row < m) {
               As[threadIdx.y][threadIdx.x] = a[Row * lda + i * TILE_DIM + threadIdx.x];
          } else {
               As[threadIdx.y][threadIdx.x] = T(0);
          }
          if (i * TILE_DIM + threadIdx.y < k && Col < n) {
               Bs[threadIdx.y][threadIdx.x] = b[(i * TILE_DIM + threadIdx.y) * ldb + Col];
          } else {
               Bs[threadIdx.y][threadIdx.x] = T(0);
          }
          __syncthreads();

          for (int n = 0; n < TILE_DIM; ++n) {
               T mul = As[threadIdx.y][n]*Bs[n][threadIdx.x];
               local_accumulator = bf16_add_rz(local_accumulator, mul);
          }
          if (i % accumulate_step == 0) {
               value = bf16_add_rz(local_accumulator, value);
               local_accumulator = T(0);
          }
          __syncthreads();
     }
     // if remaining elements are less than 64
     if (local_accumulator != T(0)) {
          value = bf16_add_rz(local_accumulator, value);
     }
     if (Row < m && Col < n) {
          c[((blockIdx.y * blockDim.y + threadIdx.y) * ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
     }
}
template __global__ void gemm_bf16_accumulate_rz_trunksize_64<float>(size_t m, size_t n, size_t k,
     const float *a, size_t lda, const float *b, size_t ldb,
     float *c, size_t ldc);
template __global__ void gemm_bf16_accumulate_rz_trunksize_64<int32>(size_t m, size_t n, size_t k,
     const int32 *a, size_t lda, const int32 *b, size_t ldb,
     int32 *c, size_t ldc);

template <typename T>
__global__ void gemm_bf16_accumulate_trunksize_64(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc) {
     T value(0);
     int Row = blockIdx.y * TILE_DIM + threadIdx.y;
     int Col = blockIdx.x * TILE_DIM + threadIdx.x;
     __shared__ T As[TILE_DIM][TILE_DIM];
     __shared__ T Bs[TILE_DIM][TILE_DIM];
     int accumulate_step = 64/TILE_DIM;
     T local_accumulator(0);
     for (int i = 0; i < (TILE_DIM + k - 1) / TILE_DIM; ++i) {
          if (i * TILE_DIM + threadIdx.x < k && Row < m) {
               As[threadIdx.y][threadIdx.x] = a[Row * lda + i * TILE_DIM + threadIdx.x];
          } else {
               As[threadIdx.y][threadIdx.x] = T(0);
          }
          if (i * TILE_DIM + threadIdx.y < k && Col < n) {
               Bs[threadIdx.y][threadIdx.x] = b[(i * TILE_DIM + threadIdx.y) * ldb + Col];
          } else {
               Bs[threadIdx.y][threadIdx.x] = T(0);
          }
          __syncthreads();

          for (int n = 0; n < TILE_DIM; ++n) {
               T mul = As[threadIdx.y][n]*Bs[n][threadIdx.x];
               local_accumulator = bf16_add(local_accumulator, mul);
          }
          if (i % accumulate_step == 0) {
               value = bf16_add(local_accumulator, value);
               local_accumulator = T(0);
          }
          __syncthreads();
     }
     // if remaining elements are less than 64
     if (local_accumulator != T(0)) {
          value = bf16_add(local_accumulator, value);
     }
     if (Row < m && Col < n) {
          c[((blockIdx.y * blockDim.y + threadIdx.y) * ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
     }
}
template __global__ void gemm_bf16_accumulate_trunksize_64<float>(size_t m, size_t n, size_t k,
     const float *a, size_t lda, const float *b, size_t ldb,
     float *c, size_t ldc);
template __global__ void gemm_bf16_accumulate_trunksize_64<int32>(size_t m, size_t n, size_t k,
     const int32 *a, size_t lda, const int32 *b, size_t ldb,
     int32 *c, size_t ldc);

template <typename T>
__global__ void gemm_accumulate_trunksize_16(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc) {
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
          T tileSum(0);
         for (int n = 0; n < TILE_DIM; ++n){
            // use am simulator
            T mul = As[threadIdx.y][n]*Bs[n][threadIdx.x];
            // Accumulate the result
            tileSum = fp32_add(tileSum, mul);
         }
           value = fp32_add(value, tileSum);
         __syncthreads();
    }
    
    if (Row < m && Col < n) {
         c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
    }
}
template __global__ void gemm_accumulate_trunksize_16<float>(size_t m, size_t n, size_t k,
   const float *a, size_t lda, const float *b, size_t ldb,
   float *c, size_t ldc);
template __global__ void gemm_accumulate_trunksize_16<int32>(size_t m, size_t n, size_t k,
     const int32 *a, size_t lda, const int32 *b, size_t ldb,
     int32 *c, size_t ldc);
template <typename T>
__global__ void gemm_accumulate_trunksize_32(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc) {
     T value(0);
     
     int Row = blockIdx.y*TRUNK_DIM_32 + threadIdx.y;
     int Col = blockIdx.x*TRUNK_DIM_32 + threadIdx.x;
     
     __shared__ T As[TRUNK_DIM_32][TRUNK_DIM_32];
     __shared__ T Bs[TRUNK_DIM_32][TRUNK_DIM_32];
     
     for (int i = 0; i < (TRUNK_DIM_32 + k - 1)/TRUNK_DIM_32; ++i) {
     
           if (i*TRUNK_DIM_32 + threadIdx.x < k && Row < m){
                 As[threadIdx.y][threadIdx.x] = a[Row*lda + i*TRUNK_DIM_32 + threadIdx.x];
           }
           else{
                 As[threadIdx.y][threadIdx.x] = T(0);
           }
     
           if (i*TRUNK_DIM_32 + threadIdx.y < k && Col < n){
                 Bs[threadIdx.y][threadIdx.x] = b[(i*TRUNK_DIM_32 + threadIdx.y)*ldb + Col];
           }
           else{
                 Bs[threadIdx.y][threadIdx.x] = T(0);
           }
     
           __syncthreads();
            T tileSum(0);
           for (int n = 0; n < TRUNK_DIM_32; ++n){
               // use am simulator
               T mul = As[threadIdx.y][n]*Bs[n][threadIdx.x];
               // Accumulate the result
               tileSum = fp32_add(tileSum, mul);
           }
             value = fp32_add(value, tileSum);
           __syncthreads();
     }
     
     if (Row < m && Col < n) {
           c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
     }
   }
template __global__ void gemm_accumulate_trunksize_32<float>(size_t m, size_t n, size_t k,
     const float *a, size_t lda, const float *b, size_t ldb,
     float *c, size_t ldc);
template __global__ void gemm_accumulate_trunksize_32<int32>(size_t m, size_t n, size_t k,
     const int32 *a, size_t lda, const int32 *b, size_t ldb,
     int32 *c, size_t ldc);
template <typename T>
__global__ void gemm_accumulate_trunksize_64(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc) {
     T value(0);
     
     int Row = blockIdx.y*TILE_DIM + threadIdx.y;
     int Col = blockIdx.x*TILE_DIM + threadIdx.x;
     
     __shared__ T As[TILE_DIM][TILE_DIM];
     __shared__ T Bs[TILE_DIM][TILE_DIM];
     
     int accumulate_step = 64/TILE_DIM;
     T local_accumulator(0);
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
               T mul = As[threadIdx.y][n]*Bs[n][threadIdx.x];
               // Accumulate the result
               local_accumulator = fp32_add(local_accumulator, mul);
           }
           if (i % accumulate_step == 0) {
               value = fp32_add(local_accumulator, value);
               local_accumulator = T(0);
           }
           __syncthreads();
     }
     
     // if remaining elements are less than 64
     if (local_accumulator != T(0)) {
           value = fp32_add(local_accumulator, value);
     }
     if (Row < m && Col < n) {
           c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
     }
   }
template __global__ void gemm_accumulate_trunksize_64<float>(size_t m, size_t n, size_t k,
     const float *a, size_t lda, const float *b, size_t ldb,
     float *c, size_t ldc);
template __global__ void gemm_accumulate_trunksize_64<int32>(size_t m, size_t n, size_t k,
     const int32 *a, size_t lda, const int32 *b, size_t ldb,
     int32 *c, size_t ldc);

template <typename T>
__global__ void gemm_fp16_accumulate_rz_trunksize_x(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc, size_t x) {
     T value(0);
     
     int Row = blockIdx.y*TILE_DIM + threadIdx.y;
     int Col = blockIdx.x*TILE_DIM + threadIdx.x;
     
     __shared__ T As[TILE_DIM][TILE_DIM];
     __shared__ T Bs[TILE_DIM][TILE_DIM];
     
     int accumulate_step = 0;
     T local_accumulator(0);
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
               T mul = As[threadIdx.y][n]*Bs[n][threadIdx.x];
               // Accumulate the result
               local_accumulator = half_add_rz(local_accumulator, mul);
               accumulate_step+=1;
               if (accumulate_step == x) {
                   value = half_add_rz(local_accumulator, value);
                   local_accumulator = T(0);
                   accumulate_step = 0;
               }
           }
           __syncthreads();
     }
     // if remaining elements are less than x
     if (local_accumulator != T(0)) {
           value = half_add_rz(local_accumulator, value);
     }

     if (Row < m && Col < n) {
           c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
     }

}
template __global__ void gemm_fp16_accumulate_rz_trunksize_x<float>(size_t m, size_t n, size_t k,
     const float *a, size_t lda, const float *b, size_t ldb,
     float *c, size_t ldc, size_t x);
template __global__ void gemm_fp16_accumulate_rz_trunksize_x<int32>(size_t m, size_t n, size_t k,
     const int32 *a, size_t lda, const int32 *b, size_t ldb,
     int32 *c, size_t ldc, size_t x);

template <typename T>
__global__ void gemm_accumulate_trunksize_x(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc, size_t x) {
     T value(0);
     
     int Row = blockIdx.y*TILE_DIM + threadIdx.y;
     int Col = blockIdx.x*TILE_DIM + threadIdx.x;
     
     __shared__ T As[TILE_DIM][TILE_DIM];
     __shared__ T Bs[TILE_DIM][TILE_DIM];
     
     int accumulate_step = 0;
     T local_accumulator(0);
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
               T mul = As[threadIdx.y][n]*Bs[n][threadIdx.x];
               // Accumulate the result
               local_accumulator = fp32_add(local_accumulator, mul);
               accumulate_step+=1;
               if (accumulate_step == x) {
                   value = fp32_add(local_accumulator, value);
                   local_accumulator = T(0);
                   accumulate_step = 0;
               }
           }
           __syncthreads();
     }
     // if remaining elements are less than x
     if (local_accumulator != T(0)) {
           value = fp32_add(local_accumulator, value);
     }
     if (Row < m && Col < n) {
           c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
     }
   }
template __global__ void gemm_accumulate_trunksize_x<float>(size_t m, size_t n, size_t k,
     const float *a, size_t lda, const float *b, size_t ldb,
     float *c, size_t ldc, size_t x);
template __global__ void gemm_accumulate_trunksize_x<int32>(size_t m, size_t n, size_t k,
     const int32 *a, size_t lda, const int32 *b, size_t ldb,
     int32 *c, size_t ldc, size_t x);


template <typename T>
__global__ void sea_gemm_accumulate_trunksize_x(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc, size_t x) {
     T value(0);
     
     int Row = blockIdx.y*TILE_DIM + threadIdx.y;
     int Col = blockIdx.x*TILE_DIM + threadIdx.x;
     
     __shared__ T As[TILE_DIM][TILE_DIM];
     __shared__ T Bs[TILE_DIM][TILE_DIM];
     
     int accumulate_step = 0;
     T positive_accumulator(0);
     T negative_accumulator(0);
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
               T mul = As[threadIdx.y][n]*Bs[n][threadIdx.x];
               // get sign of mul
               bool negative = mul < T(0);
               positive_accumulator = !negative ? fp32_add(positive_accumulator, mul) : positive_accumulator;
               negative_accumulator = negative ? fp32_add(negative_accumulator, mul) : negative_accumulator;
               accumulate_step+=1;
               if (accumulate_step == x) {
                   T pos_neg_sum = fp32_add(positive_accumulator, negative_accumulator);
                   value = fp32_add(pos_neg_sum, value);
                   positive_accumulator = T(0);
                   negative_accumulator = T(0);
                   accumulate_step = 0;
               }
           }
           __syncthreads();
     }
     // if remaining elements are less than x
     if (accumulate_step != 0) {
           T pos_neg_sum = fp32_add(positive_accumulator, negative_accumulator);
           value = fp32_add(pos_neg_sum, value);
     }
     if (Row < m && Col < n) {
           c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
     }
   }
template __global__ void sea_gemm_accumulate_trunksize_x<float>(size_t m, size_t n, size_t k,
     const float *a, size_t lda, const float *b, size_t ldb,
     float *c, size_t ldc, size_t x);
template __global__ void sea_gemm_accumulate_trunksize_x<int32>(size_t m, size_t n, size_t k,
     const int32 *a, size_t lda, const int32 *b, size_t ldb,
     int32 *c, size_t ldc, size_t x);
template <typename T>
__global__ void sea_gemm_fp16_accumulate_rz_trunksize_x(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc, size_t x) {
     T value(0);
     
     int Row = blockIdx.y*TILE_DIM + threadIdx.y;
     int Col = blockIdx.x*TILE_DIM + threadIdx.x;
     
     __shared__ T As[TILE_DIM][TILE_DIM];
     __shared__ T Bs[TILE_DIM][TILE_DIM];
     
     int accumulate_step = 0;
     T positive_accumulator = T(0);
     T negative_accumulator = T(0);
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
               T mul = As[threadIdx.y][n]*Bs[n][threadIdx.x];
               // get sign of mul
               bool negative = mul < T(0);
               positive_accumulator = !negative ? half_add_rz(positive_accumulator, mul) : positive_accumulator;
               negative_accumulator = negative ? half_add_rz(negative_accumulator, mul) : negative_accumulator;
               accumulate_step+=1;
               if (accumulate_step == x) {
                   T pos_neg_sum = half_add_rz(positive_accumulator, negative_accumulator);
                   value = half_add_rz(pos_neg_sum, value);
                   positive_accumulator = T(0);
                   negative_accumulator = T(0);
                   accumulate_step = 0;
               }
           }
           __syncthreads();
     }
     // if remaining elements are less than x
     if (accumulate_step != 0) {
           T pos_neg_sum = half_add_rz(positive_accumulator, negative_accumulator);
           value = half_add_rz(pos_neg_sum, value);
     }
     if (Row < m && Col < n) {
           c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
     }
     }
template __global__ void sea_gemm_fp16_accumulate_rz_trunksize_x<float>(size_t m, size_t n, size_t k,
     const float *a, size_t lda, const float *b, size_t ldb,
     float *c, size_t ldc, size_t x);
template __global__ void sea_gemm_fp16_accumulate_rz_trunksize_x<int32>(size_t m, size_t n, size_t k,
     const int32 *a, size_t lda, const int32 *b, size_t ldb,
     int32 *c, size_t ldc, size_t x);

template <typename T>
__global__ void sea_gemm_accumulate(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc) {     
     int Row = blockIdx.y*TILE_DIM + threadIdx.y;
     int Col = blockIdx.x*TILE_DIM + threadIdx.x;
     
     __shared__ T As[TILE_DIM][TILE_DIM];
     __shared__ T Bs[TILE_DIM][TILE_DIM];
     T positive_accumulator(0);
     T negative_accumulator(0);
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
     

           for (int n = 0; n < TILE_DIM; ++n){
               T mul = As[threadIdx.y][n]*Bs[n][threadIdx.x];
               // get sign of mul
               bool negative = mul < T(0);
               positive_accumulator = !negative ? fp32_add(positive_accumulator, mul) : positive_accumulator;
               negative_accumulator = negative ? fp32_add(negative_accumulator, mul) : negative_accumulator;
           }

           __syncthreads();
     }
     
     if (Row < m && Col < n) {
           c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = fp32_add(positive_accumulator, negative_accumulator);
     }
   }
template __global__ void sea_gemm_accumulate<float>(size_t m, size_t n, size_t k,
     const float *a, size_t lda, const float *b, size_t ldb,
     float *c, size_t ldc);
template __global__ void sea_gemm_accumulate<int32>(size_t m, size_t n, size_t k,
     const int32 *a, size_t lda, const int32 *b, size_t ldb,
     int32 *c, size_t ldc);

template <typename T>
__global__ void sea_gemm_fp16_accumulate_rz(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc) {     
     int Row = blockIdx.y*TILE_DIM + threadIdx.y;
     int Col = blockIdx.x*TILE_DIM + threadIdx.x;
     
     __shared__ T As[TILE_DIM][TILE_DIM];
     __shared__ T Bs[TILE_DIM][TILE_DIM];
     T positive_accumulator(0);
     T negative_accumulator(0);
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
     

           for (int n = 0; n < TILE_DIM; ++n){
               T mul = As[threadIdx.y][n]*Bs[n][threadIdx.x];
               // get sign of mul
               bool negative = mul < T(0);
               positive_accumulator = !negative ? half_add_rz(positive_accumulator, mul) : positive_accumulator;
               negative_accumulator = negative ? half_add_rz(negative_accumulator, mul) : negative_accumulator;
           }

           __syncthreads();
     }
     
     if (Row < m && Col < n) {
           c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = half_add_rz(positive_accumulator, negative_accumulator);
     }
   }
template __global__ void sea_gemm_fp16_accumulate_rz<float>(size_t m, size_t n, size_t k,
     const float *a, size_t lda, const float *b, size_t ldb,
     float *c, size_t ldc);
template __global__ void sea_gemm_fp16_accumulate_rz<int32>(size_t m, size_t n, size_t k,
     const int32 *a, size_t lda, const int32 *b, size_t ldb,
     int32 *c, size_t ldc);

template <typename T>
__global__ void sea_gemm_bf16_accumulate_rz(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc) {

     int Row = blockIdx.y*TILE_DIM + threadIdx.y;
     int Col = blockIdx.x*TILE_DIM + threadIdx.x;
     
     __shared__ T As[TILE_DIM][TILE_DIM];
     __shared__ T Bs[TILE_DIM][TILE_DIM];
     
     T positive_accumulator(0);
     T negative_accumulator(0);
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
     

           for (int n = 0; n < TILE_DIM; ++n){
               T mul = As[threadIdx.y][n]*Bs[n][threadIdx.x];
               // get sign of mul
               bool negative = mul < T(0);
               positive_accumulator = !negative ? bf16_add_rz(positive_accumulator, mul) : positive_accumulator;
               negative_accumulator = negative ? bf16_add_rz(negative_accumulator, mul) : negative_accumulator;
           }

           __syncthreads();
     }
     
     if (Row < m && Col < n) {
           c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = bf16_add_rz(positive_accumulator, negative_accumulator);
     }
   }
template __global__ void sea_gemm_bf16_accumulate_rz<float>(size_t m, size_t n, size_t k,
     const float *a, size_t lda, const float *b, size_t ldb,
     float *c, size_t ldc);
template __global__ void sea_gemm_bf16_accumulate_rz<int32>(size_t m, size_t n, size_t k,
     const int32 *a, size_t lda, const int32 *b, size_t ldb,
     int32 *c, size_t ldc);

template <typename T>
__global__ void sea_gemm_bf16_accmulate(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc) {
     
          int Row = blockIdx.y*TILE_DIM + threadIdx.y;
          int Col = blockIdx.x*TILE_DIM + threadIdx.x;
          
          __shared__ T As[TILE_DIM][TILE_DIM];
          __shared__ T Bs[TILE_DIM][TILE_DIM];
          T positive_accumulator(0);
          T negative_accumulator(0);
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
          
     
               for (int n = 0; n < TILE_DIM; ++n){
                    T mul = As[threadIdx.y][n]*Bs[n][threadIdx.x];
                    // get sign of mul
                    bool negative = mul < T(0);
                    positive_accumulator = !negative ? bf16_add(positive_accumulator, mul) : positive_accumulator;
                    negative_accumulator = negative ? bf16_add(negative_accumulator, mul) : negative_accumulator;
               }
     
               __syncthreads();
          }
          
          if (Row < m && Col < n) {
               c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = bf16_add(positive_accumulator, negative_accumulator);
          }
     }
template __global__ void sea_gemm_bf16_accmulate<float>(size_t m, size_t n, size_t k,
     const float *a, size_t lda, const float *b, size_t ldb,
     float *c, size_t ldc);
template __global__ void sea_gemm_bf16_accmulate<int32>(size_t m, size_t n, size_t k,
     const int32 *a, size_t lda, const int32 *b, size_t ldb,
     int32 *c, size_t ldc);

/* trunk-based accumulation */
template <typename T>
__global__ void sea_gemm_bf16_accumulate_trunksize_x(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc, size_t x) {
     T value(0);
     
     int Row = blockIdx.y*TILE_DIM + threadIdx.y;
     int Col = blockIdx.x*TILE_DIM + threadIdx.x;
     
     __shared__ T As[TILE_DIM][TILE_DIM];
     __shared__ T Bs[TILE_DIM][TILE_DIM];
     
     int accumulate_step = 0;
     T positive_accumulator(0);
     T negative_accumulator(0);
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
               T mul = As[threadIdx.y][n]*Bs[n][threadIdx.x];
               // get sign of mul
               bool negative = mul < T(0);
               positive_accumulator = !negative ? bf16_add(positive_accumulator, mul) : positive_accumulator;
               negative_accumulator = negative ? bf16_add(negative_accumulator, mul) : negative_accumulator;
               accumulate_step+=1;
               if (accumulate_step == x) {
                   T pos_neg_sum = bf16_add(positive_accumulator, negative_accumulator);
                   value = bf16_add(pos_neg_sum, value);
                   positive_accumulator = T(0);
                   negative_accumulator = T(0);
                   accumulate_step = 0;
               }
           }
           __syncthreads();
     }
     // if remaining elements are less than x
     if (accumulate_step != 0) {
           T pos_neg_sum = bf16_add(positive_accumulator, negative_accumulator);
           value = bf16_add(pos_neg_sum, value);
     }
     if (Row < m && Col < n) {
           c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
     }
}
template __global__ void sea_gemm_bf16_accumulate_trunksize_x<float>(size_t m, size_t n, size_t k,
     const float *a, size_t lda, const float *b, size_t ldb,
     float *c, size_t ldc, size_t x);
template __global__ void sea_gemm_bf16_accumulate_trunksize_x<int32>(size_t m, size_t n, size_t k,
     const int32 *a, size_t lda, const int32 *b, size_t ldb,
     int32 *c, size_t ldc, size_t x);

template <typename T>
__global__ void sea_gemm_bf16_accumulate_rz_trunksize_x(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc, size_t x) {
     T value(0);
     
     int Row = blockIdx.y*TILE_DIM + threadIdx.y;
     int Col = blockIdx.x*TILE_DIM + threadIdx.x;
     
     __shared__ T As[TILE_DIM][TILE_DIM];
     __shared__ T Bs[TILE_DIM][TILE_DIM];
     
     int accumulate_step = 0;
     T positive_accumulator(0);
     T negative_accumulator(0);
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
               T mul = As[threadIdx.y][n]*Bs[n][threadIdx.x];
               // get sign of mul
               bool negative = mul < T(0);
               positive_accumulator = !negative ? bf16_add_rz(positive_accumulator, mul) : positive_accumulator;
               negative_accumulator = negative ? bf16_add_rz(negative_accumulator, mul) : negative_accumulator;
               accumulate_step+=1;
               if (accumulate_step == x) {
                   T pos_neg_sum = bf16_add_rz(positive_accumulator, negative_accumulator);
                   value = bf16_add_rz(pos_neg_sum, value);
                   positive_accumulator = T(0);
                   negative_accumulator = T(0);
                   accumulate_step = 0;
               }
           }
           __syncthreads();
     }
     // if remaining elements are less than x
     if (accumulate_step != 0) {
           T pos_neg_sum = bf16_add_rz(positive_accumulator, negative_accumulator);
           value = bf16_add_rz(pos_neg_sum, value);
     }
     if (Row < m && Col < n) {
           c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
     }
}
template __global__ void sea_gemm_bf16_accumulate_rz_trunksize_x<float>(size_t m, size_t n, size_t k,
     const float *a, size_t lda, const float *b, size_t ldb,
     float *c, size_t ldc, size_t x);
template __global__ void sea_gemm_bf16_accumulate_rz_trunksize_x<int32>(size_t m, size_t n, size_t k,
     const int32 *a, size_t lda, const int32 *b, size_t ldb,
     int32 *c, size_t ldc, size_t x);

template <typename T>
__global__ void gemm_bf16_accumulate_trunksize_x(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc, size_t x) {
          T value(0);
     
     int Row = blockIdx.y*TILE_DIM + threadIdx.y;
     int Col = blockIdx.x*TILE_DIM + threadIdx.x;
     
     __shared__ T As[TILE_DIM][TILE_DIM];
     __shared__ T Bs[TILE_DIM][TILE_DIM];
     
     int accumulate_step = 0;
     T local_accumulator(0);
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
               T mul = As[threadIdx.y][n]*Bs[n][threadIdx.x];
               // Accumulate the result
               local_accumulator = bf16_add(local_accumulator, mul);
               accumulate_step+=1;
               if (accumulate_step == x) {
                   value = bf16_add(local_accumulator, value);
                   local_accumulator = T(0);
                   accumulate_step = 0;
               }
           }
           __syncthreads();
     }
     // if remaining elements are less than x
     if (local_accumulator != T(0)) {
           value = bf16_add(local_accumulator, value);
     }

     if (Row < m && Col < n) {
           c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
     }
}
template __global__ void gemm_bf16_accumulate_trunksize_x<float>(size_t m, size_t n, size_t k,
     const float *a, size_t lda, const float *b, size_t ldb,
     float *c, size_t ldc, size_t x);
template __global__ void gemm_bf16_accumulate_trunksize_x<int32>(size_t m, size_t n, size_t k,
     const int32 *a, size_t lda, const int32 *b, size_t ldb,
     int32 *c, size_t ldc, size_t x);
template <typename T>
__global__ void gemm_bf16_accumulate_rz_trunksize_x(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc, size_t x) {
          T value(0);
     
     int Row = blockIdx.y*TILE_DIM + threadIdx.y;
     int Col = blockIdx.x*TILE_DIM + threadIdx.x;
     
     __shared__ T As[TILE_DIM][TILE_DIM];
     __shared__ T Bs[TILE_DIM][TILE_DIM];
     
     int accumulate_step = 0;
     T local_accumulator(0);
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
               T mul = As[threadIdx.y][n]*Bs[n][threadIdx.x];
               // Accumulate the result
               local_accumulator = bf16_add_rz(local_accumulator, mul);
               accumulate_step+=1;
               if (accumulate_step == x) {
                   value = bf16_add_rz(local_accumulator, value);
                   local_accumulator = T(0);
                   accumulate_step = 0;
               }
           }
           __syncthreads();
     }
     // if remaining elements are less than x
     if (local_accumulator != T(0)) {
           value = bf16_add_rz(local_accumulator, value);
     }

     if (Row < m && Col < n) {
           c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
     }
}
template __global__ void gemm_bf16_accumulate_rz_trunksize_x<float>(size_t m, size_t n, size_t k,
     const float *a, size_t lda, const float *b, size_t ldb,
     float *c, size_t ldc, size_t x);
template __global__ void gemm_bf16_accumulate_rz_trunksize_x<int32>(size_t m, size_t n, size_t k,
     const int32 *a, size_t lda, const int32 *b, size_t ldb,
     int32 *c, size_t ldc, size_t x);
/* end of new implementation*/