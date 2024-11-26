#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <bitset>
#include <cstdint>


#define E4M3_EXP_BITS 4
#define E4M3_EXP_BIAS 7 // 2^(4-1) - 1 
#define E4M3_MAN_BITS 3
#define E4M3_MAX_EXP 7 // E4M3_EXP_BIAS
#define E4M3_MIN_EXP -6 // 1 - E4M3_EXP_BIAS
// #define E4M3_MAX_FINITE 240  // 0b11110000

// Helper union for bit manipulation
union Float32Bits {
    float f;
    uint32_t u;
};
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

// FP32 special cases
// Special Case	            Binary Representation	            Hexadecimal Representation	    Description
// Positive Zero (+0.0)	    0 00000000 00000000000000000000000	        0x00000000	            Zero with positive sign
// Negative Zero (-0.0)	    1 00000000 00000000000000000000000	        0x80000000	            Zero with negative sign
// Positive Infinity (+∞)	0 11111111 00000000000000000000000	        0x7F800000	            Positive Infinity (Not used in e4m3)
// Negative Infinity (-∞)	1 11111111 00000000000000000000000	        0xFF800000	            Negative Infinity (Not used in e4m3)
// Quiet NaN (qNaN)	        0 11111111 10000000000000000000000	        0x7FC00000	            Quiet NaN
// Signalling NaN (sNaN)	0 11111111 00000000000000000000001	        0x7F800001	            Signaling NaN

// Positive Subnormal	    0 00000000 00000000000000000000001	        0x00000001	            Smallest positive subnormal
// Negative Subnormal	    1 00000000 00000000000000000000001	        0x80000001	            Smallest negative subnormal
// Max Normal	            0 1110 110 00000000000000000000000	        0x77800000	            Largest positive normal number
// My Normal	            0 0001 000 00000000000000000000000	        0x33800000	            Smallest positive normal number
// NaN Example	            0 11111111 00000000000000000000001	        0x7F800001	            Specific NaN representation
// Another Subnormal	    0 0000 001 00000000000000000000000	        0x00020000	            Another positive subnormal number
// Special Case Example	    0 1111 101 00000000000000000000000	        0x7FA00000	            Number smaller than Max Normal but specific
uint8_t fp32_to_e4m3(float f) {
    Float32Bits f_bits;
    f_bits.f = f;
    uint32_t bits = f_bits.u;

    uint32_t sign = (bits >> 31); // Sign bit
    int32_t exponent = ((bits >> 23) & 0xFF) - 127;  // Unbiased exponent
    uint32_t unbiassed_exponent = (bits >> 23) & 0xFF;
    uint32_t mantissa = bits & 0x7FFFFF;             // Mantissa (23 bits)

    uint8_t fp8_bits = 0;
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
// e4m3 to fp32
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
float e4m3_to_fp32(uint8_t fp8_val) {
    // get sign of e4m3
    uint8_t sign = (fp8_val >> 7);
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
    
uint8_t fp32_to_e4m3_old(float f) {
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

//main function, generate random float values, convert them to e4m3 and back to float and compare new fp32_e4m3 with fp32_e4m3_old
int main() {
    // Initialize random number generator for normal distribution (mean=0, stddev=1)
    std::mt19937 rng;
    // fix random seed for reproducibility using rng.seed(0)
    rng.seed(0);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    #define TESTNUM 1000
    // Initialize matrices with random values
    std::vector<float> matA(TESTNUM);
    std::vector<float> matA_e4m3_old(TESTNUM);
    std::vector<float> matA_e4m3_new(TESTNUM);
    for (int i = 0; i < TESTNUM; ++i) {
        matA[i] = dist(rng);
        matA_e4m3_new[i] = e4m3_to_fp32(fp32_to_e4m3(matA[i]));
        matA_e4m3_old[i] = e4m3_to_fp32(fp32_to_e4m3_old(matA[i]));
    }
    // compare matA_fp32 with matA
    for (int i = 0; i < TESTNUM; ++i) {
        // print matA, matA_e4m3_old, matA_e4m3_new
        std::cout << "matA[" << i << "]: " << matA[i] << std::endl;
        std::cout << "matA_e4m3_old[" << i << "]: " << matA_e4m3_old[i] << std::endl;
        std::cout << "matA_e4m3_new[" << i << "]: " << matA_e4m3_new[i] << std::endl;
        // print binary representation of matA, matA_e4m3_old, matA_e4m3_new
        Float32Bits fb;
        fb.f = matA[i];
        std::cout << "matA[" << i << "] binary: " << std::bitset<32>(fb.u) << std::endl;
        fb.f = matA_e4m3_old[i];
        std::cout << "matA_e4m3_old[" << i << "] binary: " << std::bitset<32>(fb.u) << std::endl;
        fb.f = matA_e4m3_new[i];
        std::cout << "matA_e4m3_new[" << i << "] binary: " << std::bitset<32>(fb.u) << std::endl;

        // print space
        std::cout << std::endl;
    }
    std::cerr << "Conversion successful." << std::endl;
    return EXIT_SUCCESS;
}