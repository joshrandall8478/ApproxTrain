#ifndef ACCUMULATE_CUH
#define ACCUMULATE_CUH
#include <cuda.h>
#include <cuda_fp16.h>

__device__ __forceinline__ float fp32_add(float val, float b) {
    return val + b;
}
__device__ __forceinline__ float fp32_add_rz(float val, float b) {
    return __fadd_rz(val, b);
}
__device__ __forceinline__ float half_add(float val, float b) {
    float ret = val + b;
    return __half2float(__float2half_rn(ret));
}
__device__ __forceinline__ float half_add_rz(float val, float b) {
    float ret = val + b;
    return __half2float(__float2half_rz(ret));
    // return val + b;
}
__device__ __forceinline__ float bf16_add(float val, float b) {
    // this is a bf16 addition simulation
    // first perform fp32 addition in rne mode
    // then we  manually round down to bf16 with rne
    // special case INF/NAN is not handled here
    float ret = __fadd_rn(val, b);
    uint32_t ret_uint = __float_as_uint(ret);
    uint32_t sign = ret_uint & 0x80000000;
    uint32_t exponent = ret_uint & 0x7f800000;
    uint32_t rounding_bits = ret_uint & 0x0000fe00;
    uint32_t mant = (ret_uint & 0x007f0000) | 0x00800000;
    if (rounding_bits > 0x00008000) {
        mant += 0x00010000;
        if (mant & 0x00800000 == 0) {
            exponent += 0x00800000;
            mant = mant >> 1;
        }
    }
    uint32_t ret_uint_rounded = sign | exponent | (mant&0x007f0000);
    return __uint_as_float(ret_uint_rounded);
}
__device__ __forceinline__ float bf16_add_rz(float val, float b) {
    float ret = __fadd_rz(val, b);
    return __uint_as_float(__float_as_uint(ret) & 0xffff0000);
}


#endif // ACCUMULATE_CUH