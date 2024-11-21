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



__device__ __forceinline__ uint8_t fp32_to_e4m3(float f) {
    Float32Bits f_bits;
    f_bits.f = f;
    uint32_t bits = f_bits.u;

    uint32_t sign = (bits >> 31) & 0x1;
    int32_t exponent = ((bits >> 23) & 0xFF) - 127;  // Unbiased exponent
    uint32_t mantissa = bits & 0x7FFFFF;             // Mantissa (23 bits)

    uint8_t fp8_bits = 0;

    // Handle Zero
    if (exponent == -127 && mantissa == 0) {
        return static_cast<uint8_t>(sign << 7);
    }

    // Adjust exponent for FP8 e4m3
    int32_t new_exp = exponent + E4M3_EXP_BIAS;
    uint8_t exp_bits;
    uint8_t man_bits;

    // Handle Subnormal Numbers
    if (new_exp <= 0) {
        // Subnormal representation
        // In e4m3, subnormals have Exponent Bits = 0 and non-zero Mantissa
        // However, since e4m3 has only 3 Mantissa bits, we'll extract the top 3 bits from the FP32 mantissa
        // Shift mantissa to align with e4m3's mantissa bits
        // No rounding is applied
        uint32_t shifted_mantissa = mantissa >> (23 - E4M3_MAN_BITS);
        man_bits = static_cast<uint8_t>(shifted_mantissa & 0x7);
        exp_bits = 0;
    }
    // Handle Special Cases (NaN and Max Normal)
    else if (new_exp >= E4M3_MAX_EXP) {  // new_exp >=15
        exp_bits = 15;  // 0b1111
        if (mantissa != 0) {
            // NaN: Exponent=15, Mantissa=111
            man_bits = 7;  // 0b111
        }
        else {
            // Max Normal: Exponent=15, Mantissa=110
            man_bits = 6;  // 0b110
        }
    }
    // Handle Normal Numbers
    else {  // 1 <= new_exp <=14: Normal numbers
        exp_bits = static_cast<uint8_t>(new_exp & 0xF);  // Ensure it's within 4 bits

        // Extract the top 3 mantissa bits from FP32 mantissa
        // No rounding is applied; bits are truncated
        man_bits = static_cast<uint8_t>((mantissa >> (23 - E4M3_MAN_BITS)) & 0x7);
    }

    // Assemble the FP8 bits
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
////////////////////////////////////////////////////////////////////////////////
// FP8 E4M3 to FP32 Conversion
////////////////////////////////////////////////////////////////////////////////
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
