// fp8_conversion.cuh

#ifndef FP8_CONVERSION_CUH
#define FP8_CONVERSION_CUH

#include <cuda_fp16.h>
#include <stdint.h>



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
// E4M3
// Exponent Bias            7

// Zeros                    S.0000.000b
// Max subnormal            S.0000.111b=0.875 * 2^-6=1.3e-02
// Min subnormal            S.0000.001b=2-9=1.9e-03

// Infinities               N/A
// NaNs                     S.1111.111b
// Max normal               S.1111.110b=1.75 * 28=448.0
// Min normal               S.0001.000b=2^-6=1.5e-02

////////////////////////////////////////////////////////////////////////////////
// FP32 to FP8 e4m3 Conversion
//////////////////////////////////////////////////////////////////////////////
// E4M3 Format Specifications:
// - 1 Sign Bit
// - 4 Exponent Bits (Bias: 7)
// - 3 Mantissa Bits
//
// Special Cases:
// - Exponent Bits = 0, Mantissa = 0: Zero
// - Exponent Bits = 0, Mantissa != 0: Subnormal
// - Exponent Bits = 15, Mantissa = 111: NaN
// - Exponent Bits = 15, Mantissa = 110: Max Normal
// - Exponent Bits = 15, Mantissa = 101: A number smaller than the max normal but bigger than Exponent = 15 and Mantissa = 100
// - Exponent Bits = 1, Mantissa = 000: Min Normal
//
// Notes:
// - e4m3 does not represent Infinities.
// - Subnormals are represented but have limited precision.
// Constants for FP8 e4m3 format
#define E4M3_EXP_BITS 4
#define E4M3_EXP_BIAS 7
#define E4M3_MAN_BITS 3
#define E4M3_MAX_EXP 7
#define E4M3_MIN_EXP -6
#define E4M3_MAX_FINITE 240  // 0b11110000


__device__ __forceinline__ uint8_t fp32_to_e4m3(float f) {
    Float32Bits f_bits;
    f_bits.f = f;
    uint32_t bits = f_bits.u;

    uint32_t sign = (bits >> 31); // Sign bit
    int32_t exponent = ((bits >> 23) & 0xFF) - 127;  // Unbiased exponent
    uint32_t unbiassed_exponent = (bits >> 23) & 0xFF;
    uint32_t mantissa = bits & 0x7FFFFF;             // Mantissa (23 bits)
    
    uint8_t sign_bit = sign << 7;
    // Handle zero or biased exponent all 0s
    if (exponent == -127 && mantissa == 0) {
        return sign_bit;
    }
    // handle nan
    if (unbiassed_exponent == 0xFF && mantissa != 0) {
        // - Exponent Bits = 15, Mantissa = 111: NaN
        return sign_bit |static_cast<uint8_t>(0x7F);
    }
    // handle infinity, also include sign.

    if(unbiassed_exponent == 0xFF && mantissa == 0){
        //Exponent Bits = 15, Mantissa = 110: Max Normal
        return sign_bit | static_cast<uint8_t>(0x78);
    }

    // new exponent
    int32_t new_exp = exponent + E4M3_EXP_BIAS;
    // if new_exp < 0, map to 0
    uint8_t exp_bits;
    uint8_t man_bits;
    if (new_exp < 0) {
        // map to 0, underflow
        return sign_bit;
    } else if (new_exp > 15) {
        // map to max normal, overflow
        return sign_bit | static_cast<uint8_t>(0x78);
    } else {
        exp_bits = new_exp << 3;
        man_bits = mantissa >> 20;
        // if exp is all ones, then mantissa should never be equal to 111
        if (new_exp == 15) {
            man_bits &= 0x6;
        }
    }
    return sign_bit |   exp_bits | man_bits;
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
////////////////////////////////////////////////////////////////////////////////
// FP32 to FP8 e4m3 Conversion
//////////////////////////////////////////////////////////////////////////////
// E4M3 Format Specifications:
// - 1 Sign Bit
// - 4 Exponent Bits (Bias: 7)
// - 3 Mantissa Bits
//
// Special Cases:
// - Exponent Bits = 0, Mantissa = 0: Zero
// - Exponent Bits = 0, Mantissa != 0: Subnormal
// - Exponent Bits = 15, Mantissa = 111: NaN
// - Exponent Bits = 15, Mantissa = 110: Max Normal
// - Exponent Bits = 15, Mantissa = 101: A number smaller than the max normal but bigger than Exponent = 15 and Mantissa = 100
// - Exponent Bits = 1, Mantissa = 000: Min Normal
//
// Notes:
// - e4m3 does not represent Infinities.
// - Subnormals are represented but have limited precision.
__device__ __forceinline__ float e4m3_to_fp32(uint8_t fp8_val) {
    // get sign of e4m3
    uint8_t sign = (fp8_val >> 7) & 0x1;
    // get exp of e4m3
    uint8_t exponent = (fp8_val >> 3) & 0xF;
    // get mantissa
    uint8_t mantissa = fp8_val & 0x7;

    // prepare fp32
    uint32_t sign_bit = sign << 31;
    int32_t exponent_value;
    uint32_t mantissa_value;

    if (exponent == 0) {
        if (mantissa == 0) {
            // Zeros
            exponent_value = 0;
            mantissa_value = 0;
        } else {
            // Max and min subnormal number
            // the effective exponent for e4m3 is 1 - 7 = -6
            exponent_value = 127 - 6; // Adjust for bias (7) and exponent = 1
            mantissa_value = mantissa << (23 - 3); // Align mantissa to 23 bits
        }
    } else if (exponent == 0xF) {
        if (mantissa == 0x7) {
            // NaN (exponent and mantissa all ones)
            exponent_value = 0xFF;
            mantissa_value = 1 << 22; // Set the quiet NaN bit
        } else {
            // Maximum normal number
            exponent_value = (exponent - 7 + 127); // Adjust exponent bias
            mantissa_value = mantissa << (23 - 3);
        }
    } else {
        // Normalized number
        exponent_value = exponent - 7 + 127; // Adjust exponent bias
        mantissa_value = mantissa << (23 - 3); // Align mantissa to 23 bits
    }

    uint32_t result_bits = sign_bit | (exponent_value << 23) | mantissa_value;
    Float32Bits fb;
    fb.u = result_bits;
    return fb.f;
}

////////////////////////////////////////////////////////////////////////////////
// FP8 E5M2 to FP32 Conversion
////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ float e5m2_to_fp32(uint8_t fp8_val) {
    uint8_t sign = (fp8_val >> 7) & 0x1;
    uint8_t exponent = (fp8_val >> 2) & 0x1F;
    uint8_t mantissa = fp8_val & 0x3;

    uint32_t sign_bit = sign << 31;
    int32_t exponent_value;
    uint32_t mantissa_value;

    if (exponent == 0) {
        if (mantissa == 0) {
            // Zero
            exponent_value = 0;
            mantissa_value = 0;
        } else {
            // Subnormal number
            // the effective exponent for e5m2 is 1 - 15 = -14
            exponent_value = 127 - 14; // Adjust for bias (15) and exponent = 1
            mantissa_value = mantissa << (23 - 2); // Align mantissa to 23 bits
        }
    } else if (exponent == 0x1F) {
        // NaN or Infinity
        exponent_value = 0xFF;
        mantissa_value = mantissa ? (1 << 22) : 0; // NaN if mantissa != 0
    } else {
        // Normalized number
        exponent_value = exponent - 15 + 127; // Adjust exponent bias
        mantissa_value = mantissa << (23 - 2); // Align mantissa to 23 bits
    }

    uint32_t result_bits = sign_bit | (exponent_value << 23) | mantissa_value;
    Float32Bits fb;
    fb.u = result_bits;
    return fb.f;
}





#endif  // FP8_CONVERSION_CUH
