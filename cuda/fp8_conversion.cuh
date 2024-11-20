// fp8_conversion.cuh

#ifndef FP8_CONVERSION_CUH
#define FP8_CONVERSION_CUH

#include <cuda_fp16.h>
#include <stdint.h>

// Constants for FP8 e4m3 format
#define E4M3_EXP_BITS 4
#define E4M3_EXP_BIAS 7
#define E4M3_MAN_BITS 3
#define E4M3_MAX_EXP 8
#define E4M3_MIN_EXP -7
#define E4M3_MAX_FINITE 240  // 0b11110000

// Constants for FP8 e5m2 format
#define E5M2_EXP_BITS 5
#define E5M2_EXP_BIAS 15
#define E5M2_MAN_BITS 2
#define E5M2_MAX_EXP 15
#define E5M2_MIN_EXP -14
#define E5M2_MAX_FINITE 252  // 0b11111100

// Helper union for bit manipulation
union Float32Bits {
    float f;
    uint32_t u;
};

////////////////////////////////////////////////////////////////////////////////
// FP32 to FP8 e4m3 Conversion
////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ uint8_t fp32_to_e4m3(float f) {
    Float32Bits f_bits;
    f_bits.f = f;
    uint32_t bits = f_bits.u;

    uint32_t sign = (bits >> 31) & 0x1;
    int32_t exponent = ((bits >> 23) & 0xFF) - 127;  // Unbiased exponent
    uint32_t mantissa = bits & 0x7FFFFF;             // Mantissa (23 bits)

    uint8_t fp8_bits = 0;

    // Handle zero
    if (exponent == -127 && mantissa == 0) {
        return static_cast<uint8_t>(sign << 7);
    }

    // Adjust exponent for FP8 e4m3
    int32_t new_exp = exponent + E4M3_EXP_BIAS;
    uint8_t exp_bits;
    uint8_t man_bits;

    // Handle overflow (Inf)
    if (new_exp >= (1 << E4M3_EXP_BITS) - 1) {
        exp_bits = (1 << E4M3_EXP_BITS) - 1;
        man_bits = 0;
    }
    // Handle underflow (Zero)
    else if (new_exp <= 0) {
        // Subnormal numbers (not representable in FP8 e4m3)
        exp_bits = 0;
        man_bits = 0;
    }
    // Normal numbers
    else {
        exp_bits = static_cast<uint8_t>(new_exp & ((1 << E4M3_EXP_BITS) - 1));
        // Take the most significant bits from mantissa
        man_bits = static_cast<uint8_t>(mantissa >> (23 - E4M3_MAN_BITS)) & ((1 << E4M3_MAN_BITS) - 1);
    }

    fp8_bits = (sign << 7) | (exp_bits << E4M3_MAN_BITS) | man_bits;
    return fp8_bits;
}

////////////////////////////////////////////////////////////////////////////////
// FP32 to FP8 e5m2 Conversion
////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ uint8_t fp32_to_e5m2(float f) {
    Float32Bits f_bits;
    f_bits.f = f;
    uint32_t bits = f_bits.u;

    uint32_t sign = (bits >> 31) & 0x1;
    int32_t exponent = ((bits >> 23) & 0xFF) - 127;  // Unbiased exponent
    uint32_t mantissa = bits & 0x7FFFFF;             // Mantissa (23 bits)

    uint8_t fp8_bits = 0;

    // Handle zero
    if (exponent == -127 && mantissa == 0) {
        return static_cast<uint8_t>(sign << 7);
    }

    // Adjust exponent for FP8 e5m2
    int32_t new_exp = exponent + E5M2_EXP_BIAS;
    uint8_t exp_bits;
    uint8_t man_bits;

    // Handle overflow (Inf)
    if (new_exp >= (1 << E5M2_EXP_BITS) - 1) {
        exp_bits = (1 << E5M2_EXP_BITS) - 1;
        man_bits = 0;
    }
    // Handle underflow (Zero)
    else if (new_exp <= 0) {
        // Subnormal numbers (not representable in FP8 e5m2)
        exp_bits = 0;
        man_bits = 0;
    }
    // Normal numbers
    else {
        exp_bits = static_cast<uint8_t>(new_exp & ((1 << E5M2_EXP_BITS) - 1));
        // Take the most significant bits from mantissa
        man_bits = static_cast<uint8_t>(mantissa >> (23 - E5M2_MAN_BITS)) & ((1 << E5M2_MAN_BITS) - 1);
    }

    fp8_bits = (sign << 7) | (exp_bits << E5M2_MAN_BITS) | man_bits;
    return fp8_bits;
}

#endif  // FP8_CONVERSION_CUH
