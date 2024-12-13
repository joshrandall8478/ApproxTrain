#ifndef FP8_CONVERSION_H
#define FP8_CONVERSION_H
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <bitset>
#include <cstdint>
#include <cuda_fp16.h>
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
uint8_t fp8e5m2_from_fp32_value_torch(float f) {
  /*
   * Binary representation of fp32 infinity
   * 0 11111111 00000000000000000000000
   */
  constexpr uint32_t fp32_inf = UINT32_C(255) << 23;

  /*
   * Binary representation of 65536.0f, which is the first value
   * not representable in fp8e5m2 range:
   * 0 11111 00 - fp8e5m2
   * 0 10001111 00000000000000000000000 - fp32
   */
  constexpr uint32_t fp8_max = UINT32_C(143) << 23;

  /*
   * A mask for converting fp32 numbers lower than fp8e5m2 normal range
   * into denorm representation
   * magic number: ((127 - 15) + (23 - 2) + 1)
   */
  constexpr uint32_t denorm_mask = UINT32_C(134) << 23;

  uint32_t f_bits = fp32_to_bits(f);
  uint8_t result = 0u;

  /*
   * Extract the sign of the input number into the high bit of the 32-bit word:
   *
   *      +---+----------------------------------+
   *      | S |0000000 00000000 00000000 00000000|
   *      +---+----------------------------------+
   * Bits  31                 0-31
   */
  const uint32_t sign = f_bits & UINT32_C(0x80000000);

  /*
   * Set sign bit to 0
   */
  f_bits ^= sign;

  if (f_bits >= fp8_max) {
    // NaN - all exponent and mantissa bits set to 1
    result = f_bits > fp32_inf ? UINT8_C(0x7F) : UINT8_C(0x7C);
  } else {
    if (f_bits < (UINT32_C(113) << 23)) {
      // Input number is smaller than 2^(-14), which is the smallest
      // fp8e5m2 normal number
      f_bits =
          fp32_to_bits(fp32_from_bits(f_bits) + fp32_from_bits(denorm_mask));
      result = static_cast<uint8_t>(f_bits - denorm_mask);
    } else {
      // resulting mantissa is odd
      uint32_t mant_odd = (f_bits >> 21) & 1;

      // update exponent, rounding bias part 1
      f_bits += ((uint32_t)(15 - 127) << 23) + 0xFFFFF;

      // rounding bias part 2
      f_bits += mant_odd;

      // take the bits!
      result = static_cast<uint8_t>(f_bits >> 21);
    }
  }

  result |= static_cast<uint8_t>(sign >> 24);
  return result;
}
float fp8e5m2_to_fp32_value_torch(uint8_t input) {
  /*
   * Extend the fp8 E5M2 number to 32 bits and shift to the
   * upper part of the 32-bit word:
   *      +---+----+---+-----------------------------+
   *      | S |EEEEE|MM|0000 0000 0000 0000 0000 0000|
   *      +---+----+---+-----------------------------+
   * Bits  31 26-30 24-25          0-23
   *
   * S - sign bit, E - bits of the biased exponent, M - bits of the mantissa, 0
   * - zero bits.
   */
  uint16_t half_representation = input;
  half_representation <<= 8;
  return __half2float(*reinterpret_cast<__half*>(&half_representation));
}

float clip_fp8_e5m2_torch(float a){
    return fp8e5m2_to_fp32_value_torch(fp8e5m2_from_fp32_value_torch(a));
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
        if (exp < -1) {
            // The value is too small even for subnormal.
            // Flush to zero, preserving sign.
            return (uint8_t)sign;
        }

        // If the original FP32 number was normal (biased_exp != 0), add the implicit leading 1.
        if (biased_exp != 0) {
            mant |= 0x00800000u;
        } 
        // If biased_exp == 0 (FP32 subnormal), we do not add the implicit 1.

   
        int shift = 22 - exp; // shift to get the mantissa
        // if (shift > 31) shift = 31; // clamp shift to avoid overshifting

        uint32_t val = mant;
        uint32_t mant_fp8 = val >> shift;

        // Rounding half-up: check the bit below the truncated bits
        if (val & (1u << (shift - 1))) {
            mant_fp8++;
            if (mant_fp8 == 4) {
                return (uint8_t)(sign | 0x4);
            }
        }

   

        // If rounding results in zero, the value is too small
        if (mant_fp8 == 0) {
            return (uint8_t)sign;
        }

        // Construct subnormal fp8 value (exponent=0)
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
        if (mant_fp8 == 4) {
            // Overflow in mantissa
            mant_fp8 = 0;
            exp++;
            // overflow in exp
            // Handle very large exponent (overflow)
            if (exp >= (FP8_EXP_MAX - 1)) {
                // Clamp to max normal
                return (uint8_t)(sign | MAX_NORMAL_FP8);
            }
        }

    

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
        // Subnormal
        // Start with the exponent for subnormals in FP8
        int32_t e = 1 - fp8_bias; // Exponent for subnormal FP8
        out_exp = 127 + e; // Map FP8 exponent to FP32 range

        // Normalise the mantissa by finding the leading 1 position
        uint32_t frac = mantissa;
        int shift = 0;
        while ((frac & 0x4) == 0) { // Keep shifting until the MSB is aligned
            frac <<= 1;
            shift++;
        }

        // Adjust the exponent based on the number of shifts
        out_exp -= shift;

        // Place the mantissa in the FP32 fraction field
        out_mant = (frac & 0x3) << (23 - FP8_MANT_BITS);
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
    uint8_t h = fp32_to_fp8_e5m2(a, fp8_bias);
    return fp8_e5m2_to_fp32(h, fp8_bias);
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


float clip_fp8_e5m2_rtz(float a) {
    // convert to half
    __half b = __float2half(a);
    // as uint
    uint16_t h_bits = *reinterpret_cast<uint16_t*>(&b);
    // convert to fp8 e5m2
    h_bits = h_bits & 0xff00;
    // convert back to half
    b = *reinterpret_cast<__half*>(&h_bits);
    // convert back to float
    return __half2float(b);
}

float clip_fp16(float a){
    // convert to half
    __half b = __float2half(a);
    // convert back to float
    return __half2float(b);
}
#endif // FP8_CONVERSION_H