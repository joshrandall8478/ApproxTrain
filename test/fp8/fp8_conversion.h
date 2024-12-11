#ifndef FP8_CONVERSION_H
#define FP8_CONVERSION_H
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <bitset>
#include <cstdint>
union Float32Bits {
    float f;
    uint32_t u;
};

uint32_t fp32_to_bits(float f) {
    Float32Bits fb;
    fb.f = f;
    return fb.u;
}

float fp32_from_bits(uint32_t u) {
    Float32Bits fb;
    fb.u = u;
    return fb.f;
}
// Convert from fp32 to fp8 e5m2 with variable bias
// Clamps large values to max normal, flushes very small values to zero
uint8_t fp32_to_fp8_e5m2(float a, int fp8_bias) {
    const uint32_t FP32_SIGN_MASK = 0x80000000u;
    const uint32_t FP32_EXP_MASK  = 0x7F800000u;
    const uint32_t FP32_MANT_MASK = 0x007FFFFFu;
    const int      FP32_EXP_BIAS  = 127;

    // FP8 E5M2 parameters
    const int FP8_EXP_BITS   = 5;
    const int FP8_MANT_BITS  = 2;
    const int FP8_EXP_MAX    = 31;
    // Exponent=31 is for Inf/NaN. So max normal exponent code = 30
    // Format: [sign:1][exponent:5][mantissa:2]
    // Max normal: exponent=30 (11110), mantissa=3 (11) => 0x7B
    // NaN chosen: exponent=31 (11111), mantissa=3 (11) => 0x7F

    const uint8_t NAN_FP8      = 0x7F;
    const uint8_t MAX_NORMAL_FP8 = 0x7B; // Largest normal value x11110.11

    uint32_t bits = fp32_to_bits(a);
    uint32_t sign = (bits & FP32_SIGN_MASK) >> 24; // move sign to bit 7 of the fp8 byte
    uint32_t biased_exp = (bits & FP32_EXP_MASK) >> 23;
    uint32_t mant = bits & FP32_MANT_MASK;

    // Handle NaN/Inf
    if (biased_exp == 0xFF) {
        if (mant != 0) {
            // NaN
            return NAN_FP8;
        } else {
            // Infinity: clamp to max normal
            return (uint8_t)(sign | MAX_NORMAL_FP8);
        }
    }

    if (biased_exp == 0 && mant == 0) {
        // Zero
        return (uint8_t)sign; 
    }

    // Compute the unbiased exponent
    int exp = (int)biased_exp - FP32_EXP_BIAS;

    // Adjust exponent by fp8_bias
    exp += fp8_bias;

    // Handle very large exponent (overflow)
    if (exp >= (FP8_EXP_MAX - 1)) {
        // Clamp to max normal
        return (uint8_t)(sign | MAX_NORMAL_FP8);
    }

    // Handle subnormals and underflow
    // For exponent <= 0, we may have subnormals
    if (exp <= 0) {
        // For subnormal in e5m2:
        // value = sign * mantissa/(2^m) * 2^(1 - bias) 
        // If exp < - (2+1) [some margin], flush to zero.
        // The smallest exponent for normal is 1 - fp8_bias.
        // If we are too small, just return zero.
        if (exp < -((1 << FP8_MANT_BITS))) {
            return (uint8_t)sign; // Underflow to zero
        }

        // Prepare for subnormal: 
        // Add implicit leading 1 to mantissa if originally was a normal fp32 number
        mant |= 0x00800000u; 
        // Shift mantissa to fit into 2 bits after adjusting for subnormal scale
        // Each decrement in exp below 1 means dividing by 2.
        int shift = 23 - FP8_MANT_BITS + (1 - exp); 
        // (1 - exp) is how many extra powers of two we need to shift by.
        // exp ≤ 0, so (1 - exp) ≥ 1
        // Example: if exp = 0 => shift = 23 - 2 + 1 = 22 bits
        // If exp = -1 => shift = 23 - 2 + 2 = 23 bits, etc.

        // Avoid negative shifts just in case, but we already handled that by checking exp.
        if (shift > 31) shift = 31; // safety
        uint32_t mant_fp8 = mant >> shift;

        // Rounding: check next bit for rounding up
        // If the bit just below the cut is set, round up
        // This is a simple round half up strategy.
        if (shift > 0 && (mant & (1u << (shift-1)))) {
            mant_fp8++;
        }

        // Mask mantissa
        mant_fp8 &= 0x03;

        // For subnormals in FP8, exponent = 0
        uint8_t fp8 = (uint8_t)(sign | mant_fp8);
        return fp8;
    }

    // Normal number path:
    // We have 1.mant in fp32. 
    // Shift mant to get fp8 mantissa
    uint32_t mant_round_bit = 1u << (23 - FP8_MANT_BITS - 1);
    uint32_t mant_fp8 = mant >> (23 - FP8_MANT_BITS);

    // Rounding up if the next bit after truncated bits is set
    if (mant & mant_round_bit) {
        mant_fp8++;
    }
    mant_fp8 &= 0x03; // 2-bit mantissa

    // Combine sign, exponent, mantissa
    // exp is between 1 and 30 (for normal)
    uint8_t fp8 = (uint8_t)(sign | ((exp << FP8_MANT_BITS) & 0x7C) | (mant_fp8 & 0x03));
    return fp8;
}


// Convert from fp8 e5m2 to fp32
float fp8_e5m2_to_fp32(uint8_t h, int fp8_bias) {
    const int FP8_EXP_BITS   = 5;
    const int FP8_MANT_BITS  = 2;
    const int FP8_EXP_MASK   = 0x1F; // 5 bits
    const int FP8_EXP_BIAS   = fp8_bias;
    
    // Extract sign, exponent, mantissa
    uint32_t sign     = (h & 0x80) ? 1 : 0;
    uint32_t exponent = (h >> FP8_MANT_BITS) & FP8_EXP_MASK;
    uint32_t mantissa = h & ((1 << FP8_MANT_BITS) - 1); // 2 bits

    uint32_t out_sign = sign << 31;
    int32_t out_exp;
    uint32_t out_mant;
    
    if (exponent == 0x1F) {
        // Inf or NaN
        if (mantissa == 0) {
            // Inf
            out_exp = 0xFF;
            out_mant = 0x000000;
        } else {
            // NaN
            out_exp = 0xFF;
            out_mant = 0x200000; // Quiet NaN
        }
    } else if (exponent == 0) {
        // Subnormal or zero
        if (mantissa == 0) {
            // zero
            out_exp = 0; 
            out_mant = 0;
        } else {
            // subnormal
            // value = (-1)^sign * (mantissa/(2^m)) * 2^(1 - bias)
            // in fp32: out_exp = 127 + (1 - fp8_bias) - 1 (for subnormal no leading 1)
            // Actually: smallest normal exponent = 1 - bias
            // subnormal exponent in fp32 = (1 - bias) - 1 = -bias
            // So out_exp = FP32_BIAS + ((1 - FP8_BIAS) - 1) = FP32_BIAS - FP8_BIAS
            // Actually simpler: 
            // For subnormals: exponent in linear form = 1 - FP8_BIAS
            // out_exp = 127 + (1 - FP8_BIAS)
            int32_t e = 1 - fp8_bias; 
            out_exp = 127 + e; 
            // fraction = mantissa / (2^2) * 2^(e)
            // to build mantissa in fp32: (fraction = mantissa/(4)) * 2^e
            // We'll construct out_mant accordingly:
            // fraction * 2^(out_exp-127) = mantissa/(4)
            // mantissa/(4) means shift mantissa up by (23 - 2) bits for fp32
            // but we have no leading 1, so just place mantissa in mant bits:
            uint32_t frac = mantissa; 
            // place mantissa in 23-bit fraction field: frac * 2^(23-2)=frac<<21
            out_mant = frac << (23 - FP8_MANT_BITS);
            // no leading 1 in subnormals, so no extra addition
            // might need to adjust exponent if resulting fraction < 1.0
            // Actually subnormal: fraction = mantissa/(4)
            // So final: sign + out_exp + out_mant good.
        }
    } else {
        // Normal number
        // value = (-1)^sign * (1 + mantissa/(2^2)) * 2^(exponent - FP8_BIAS)
        int32_t e = (int32_t)exponent - fp8_bias;
        out_exp = 127 + e; 
        // mantissa: we have (1 + mantissa/4)
        // The fp32 mantissa is (1.mantissa_bits)
        // So for fp32: leading 1 = implicit in mantissa, 
        // place (mantissa) in the lower bits of 23-bit fraction.
        // mantissa in fp8 is 2 bits, so mantissa << (23-2) = mantissa<<21
        // plus the implicit leading 1: leading 1 in binary is at bit 23 -> out_mant = (1<<23) + (mantissa << 21)
        out_mant = (1 << 23) + (mantissa << (23 - FP8_MANT_BITS));
    }

    uint32_t out_exp_bits = ((uint32_t)out_exp & 0xFF) << 23;
    uint32_t out_bits = (out_sign & 0x80000000) | out_exp_bits | (out_mant & 0x7FFFFF);
    return fp32_from_bits(out_bits);
}
float clip_fp8_e5m2(float a, int fp8_bias=24) {
    // Direct round-trip:
    uint8_t h = fp32_to_fp8_e5m2(a, 24);
    return fp8_e5m2_to_fp32(h, 24);
}
// Convert fp32 to fp8 e4m3 with variable bias
uint8_t fp32_to_fp8_e4m3(float a, int fp8_bias) {
    const uint32_t FP32_SIGN_MASK = 0x80000000u;
    const uint32_t FP32_EXP_MASK  = 0x7F800000u;
    const uint32_t FP32_MANT_MASK = 0x007FFFFFu;
    const int      FP32_EXP_BIAS  = 127;

    // e4m3 format constants
    const int FP8_EXP_BITS  = 4;
    const int FP8_MANT_BITS = 3;
    const int FP8_EXP_MAX   = (1 << FP8_EXP_BITS) - 1; // 15

    // Special patterns
    // NaN = 0x7F (E=1111,M=111)
    // Max normal = 0x7E (E=1111,M=110), sign adjusted for negative if needed
    const uint8_t FP8_QUIET_NAN  = 0x7F;
    const uint8_t FP8_MAX_NORMAL = 0x7E;

    uint32_t bits = fp32_to_bits(a);
    uint32_t sign_bit = (bits & FP32_SIGN_MASK) >> 24; 
    uint32_t biased_exp = (bits & FP32_EXP_MASK) >> 23;
    uint32_t mant = bits & FP32_MANT_MASK;

    // Handle NaN/Inf
    if (biased_exp == 0xFF) {
        if (mant != 0) {
            // NaN
            return (uint8_t)(sign_bit | FP8_QUIET_NAN);
        } else {
            // Inf -> map to max normal
            return (uint8_t)(sign_bit | FP8_MAX_NORMAL);
        }
    }

    // Zero
    if (biased_exp == 0 && mant == 0) {
        return (uint8_t)sign_bit; 
    }

    int unbiased_exp = (int)biased_exp - FP32_EXP_BIAS;
    int e = unbiased_exp + fp8_bias;

    // Overflow -> max normal
    if (e > FP8_EXP_MAX) {
        return (uint8_t)(sign_bit | FP8_MAX_NORMAL);
    }

    // Shift for mantissa extraction
    const int M_SHIFT = (23 - FP8_MANT_BITS);

    if (e >= 1) {
        // Normal number in fp8
        uint32_t mant_fp8 = mant >> M_SHIFT;
        uint32_t rounding_bit = (mant >> (M_SHIFT - 1)) & 1;
        if (rounding_bit) mant_fp8++;
        if (mant_fp8 > 0x07) {
            // Rounding overflow
            mant_fp8 = 0x07;
            e += 1;
            if (e > FP8_EXP_MAX) {
                return (uint8_t)(sign_bit | FP8_MAX_NORMAL);
            }
        }
        uint8_t fp8 = (uint8_t)(sign_bit | ((e << FP8_MANT_BITS) & 0x78) | (mant_fp8 & 0x07));
        return fp8;
    } else {
        // Subnormal or zero
        // Check for too small values -> zero
        if (unbiased_exp < (1 - fp8_bias - FP8_MANT_BITS)) {
            return (uint8_t)sign_bit;
        }

        // Subnormal
        int shift = (1 - e) + M_SHIFT; 
        uint32_t val = (mant | 0x00800000u);
        uint32_t sub_mant = val >> shift;
        if (shift > 0 && (val & (1u << (shift - 1)))) {
            sub_mant++;
        }
        if (sub_mant > 0x07) sub_mant = 0x07;
        if (sub_mant == 0) {
            // tiny -> zero
            return (uint8_t)sign_bit;
        }
        return (uint8_t)(sign_bit | (sub_mant & 0x07));
    }
}

// Convert fp8 e4m3 to fp32 with variable bias
float fp8_e4m3_to_fp32(uint8_t h, int fp8_bias) {
    const int FP8_EXP_BITS  = 4;
    const int FP8_MANT_BITS = 3;

    uint32_t sign = (h & 0x80) ? 1 : 0;
    uint32_t exponent = (h >> FP8_MANT_BITS) & 0x0F;
    uint32_t mantissa = h & 0x07;

    uint32_t out_sign = sign << 31;
    int out_exp;
    uint32_t out_mant;

    if (exponent == 0x0F) {
        // Possibly NaN or max normal pattern
        if (mantissa == 0x07) {
            // NaN
            out_exp = 0xFF;
            out_mant = 0x200000; // quiet NaN
        } else {
            // max normal pattern (no Inf)
            int e = 15 - fp8_bias; 
            out_exp = 127 + e;
            out_mant = (1 << 23) + (mantissa << (23 - FP8_MANT_BITS));
        }
    } else if (exponent == 0) {
        if (mantissa == 0) {
            // zero
            out_exp = 0;
            out_mant = 0;
        } else {
            // subnormal
            // exponent = 1 - fp8_bias
            int e = 1 - fp8_bias;
            // subnormals have no implicit 1
            out_exp = 127 + e - 1;
            out_mant = (mantissa << (23 - FP8_MANT_BITS));
        }
    } else {
        // normal
        int e = (int)exponent - fp8_bias;
        out_exp = 127 + e;
        out_mant = (1 << 23) + (mantissa << (23 - FP8_MANT_BITS));
    }

    uint32_t out_exp_bits = ((uint32_t)out_exp & 0xFF) << 23;
    uint32_t out_bits = (out_sign & 0x80000000) | out_exp_bits | (out_mant & 0x7FFFFF);
    return fp32_from_bits(out_bits);
}

// Clipping function: fp32 -> fp8 e4m3 -> fp32 with variable bias
float clip_fp8_e4m3(float a, int fp8_bias=14) {
    uint8_t h = fp32_to_fp8_e4m3(a, fp8_bias);
    return fp8_e4m3_to_fp32(h, fp8_bias);
}
#endif // FP8_CONVERSION_H