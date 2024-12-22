#ifndef GEMM_CUH
#define GEMM_CUH
#define TILE_DIM 16
#define TRUNK_DIM_32 32
/* below kernels are intended for trunk based accumulation*/
template <typename T>
__global__ void gemm_fp16_accumulate_trunksize_16(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc);
template <typename T>
__global__ void gemm_fp16_accumulate_trunksize_32(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc);
template <typename T>
__global__ void gemm_fp16_accumulate_trunksize_64(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc);

template <typename T> 
__global__ void gemm_fp16_accumulate_rz_trunksize_16(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc);
template <typename T> 
__global__ void gemm_fp16_accumulate_rz_trunksize_32(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc);
template <typename T>
__global__ void gemm_fp16_accumulate_rz_trunksize_64(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc);
template <typename T>
__global__ void gemm_bf16_accumulate_rz_trunksize_16(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc);
template <typename T>
__global__ void gemm_bf16_accumulate_rz_trunksize_32(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc);
template <typename T>
__global__ void gemm_bf16_accumulate_rz_trunksize_64(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc);
template <typename T>
__global__ void gemm_bf16_accumulate_trunksize_16(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc);
template <typename T>
__global__ void gemm_bf16_accumulate_trunksize_32(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc);
template <typename T>
__global__ void gemm_bf16_accumulate_trunksize_64(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc);
/* these four kernels are intended for pairing with lower precision that is fp8*/
template <typename T>
__global__ void gemm_bf16_accumulate(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc
   );
template <typename T>
__global__ void gemm_bf16_accumulate_rz(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc
   );
template <typename T>
__global__ void gemm_fp16_accumulate(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc);
template <typename T>
__global__ void gemm_fp16_accumulate_rz(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc);

/* gemm non-lut fp32 */
template <typename T>
__global__ void gemm(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc
   );
template <typename T>
__global__ void gemm_rz(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc
   );
/* gemm non-lut bfloat16*/
template <typename T>
__global__ void gemm_bf16(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc
   );
// gemm_bf16_rz
template <typename T>
__global__ void gemm_bf16_rz(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc
   );
/* gemm non-lut fp16*/
template <typename T>
__global__ void gemm_fp16(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc
   );
// gemm_fp16_rz
template <typename T>
__global__ void gemm_fp16_rz(size_t m, size_t n, size_t k,
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
// gemm_rz lut_based
template <typename T>
__global__ void gemm_rz(size_t m, size_t n, size_t k,
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

#endif
