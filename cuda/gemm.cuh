#ifndef GEMM_CUH
#define GEMM_CUH

/* gemm non-lut fp32 */
template <typename T>
__global__ void gemm(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc
   );
/* gemm non-lut bfloat16*/
template <typename T>
__global__ void gemm_bf16(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc
   );
/* gemm non-lut fp16*/
template <typename T>
__global__ void gemm_fp16(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc
   );
/* gemm non-lut fp8*/
template <typename T>
__global__ void gemm_e4m3(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc
   );
template <typename T>
__global__ void gemm_e5m2(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc
   );



/* gemm lut 8-bit exponents typed e.g. Bfloat16*/
template <typename T>
__global__ void gemm(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc, cudaTextureObject_t mant_lut,
   uint32_t mant_mask, uint8_t a_shift, uint8_t b_shift, uint8_t mant_bitwidth
   );
/* gemm lut 5-bit exponents typed e.g. FP16*/
template <typename T>
__global__ void gemm_5exp(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc, cudaTextureObject_t mant_lut,
   uint32_t mant_mask, uint8_t a_shift, uint8_t b_shift, uint8_t mant_bitwidth
   );
/* gemm lut fp8 kernels*/
template <typename T>
__global__ void gemm_e4m3(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc, cudaTextureObject_t mant_lut
   );

template <typename T>
__global__ void gemm_e5m2(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc, cudaTextureObject_t mant_lut
   );
   
#endif
