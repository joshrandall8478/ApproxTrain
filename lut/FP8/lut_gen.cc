#include <cstdint>
#include <cmath>
#include <iostream>
#include <fstream>
#include <cstring>
#include <iomanip>

// Union to interpret a float as uint32_t and vice versa
union Float32Bits {
    float f;
    uint32_t u;
};
// E4M3
// Exponent Bias            7

// Zeros                    S.0000.000b
// Max subnormal            S.0000.111b=0.875 * 2^-6=1.3e-02
// Min subnormal            S.0000.001b=2-9=1.9e-03

// Infinities               N/A
// NaNs                     S.1111.111b
// Max normal               S.1111.110b=1.75 * 28=448.0
// Min normal               S.0001.000b=2^-6=1.5e-02



float e4m3_to_fp32(uint8_t fp8_val) {
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

// E5M2
// Exponent Bias	15
// Infinities	    S.11111.00b
// NaNs	            S.11111.{01, 10, 11}b

// Zeros	        S.00000.00b
// Max subnormal	S.00000.11b=0.75 * 2^-14=4.5e-05
// Min subnormal	S.00000.01b=2^-16=1.5e-05


// Max normal	    S.11110.11b=1.75 * 2^15=57344.0
// Min normal	    S.00001.00b=2-14=6.1e-05


float e5m2_to_fp32(uint8_t fp8_val) {
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


void generate_fp32_mul_lut(float* lut, bool use_e4m3) {
    for (uint16_t a = 0; a < 256; ++a) {
        for (uint16_t b = 0; b < 256; ++b) {
            uint16_t idx = (a << 8) | b;
            float a_fp32, b_fp32;

            if (use_e4m3) {
                a_fp32 = e4m3_to_fp32((uint8_t)a);
                b_fp32 = e4m3_to_fp32((uint8_t)b);
            } else {
                a_fp32 = e5m2_to_fp32((uint8_t)a);
                b_fp32 = e5m2_to_fp32((uint8_t)b);
            }

            // Multiply in FP32
            float result_fp32 = a_fp32 * b_fp32;
            std::cout << a_fp32 << b_fp32 << result_fp32 << std::endl;
            
            // Store FP32 result in LUT
            lut[idx] = result_fp32;
        }
    }
}


void save_fp32_lut_to_file(const float* lut, const char* filename) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }
    ofs.write(reinterpret_cast<const char*>(lut), 256 * 256 * 2* sizeof(float));
    ofs.close();
    if (!ofs.good()) {
        std::cerr << "Error occurred while writing to file: " << filename << std::endl;
    }
}

// // Main Function
// int main() {
//     // Allocate memory for LUTs
//     float* e4m3_lut = new float[256 * 256];
//     float* e5m2_lut = new float[256 * 256];

//     std::memset(e4m3_lut, 0, 256 * 256 * sizeof(float));
//     std::memset(e5m2_lut, 0, 256 * 256 * sizeof(float));

//     // Generate and save E4M3 FP32 LUT
//     std::cout << "Generating E4M3 FP32 multiplication LUT..." << std::endl;
//     generate_fp32_mul_lut(e4m3_lut, true);
//     save_fp32_lut_to_file(e4m3_lut, "e4m3_fp32_mul_lut.bin");

//     // Generate and save E5M2 FP32 LUT
//     std::cout << "Generating E5M2 FP32 multiplication LUT..." << std::endl;
//     generate_fp32_mul_lut(e5m2_lut, false);
//     save_fp32_lut_to_file(e5m2_lut, "e5m2_fp32_mul_lut.bin");

//     // Clean up
//     delete[] e4m3_lut;
//     delete[] e5m2_lut;

//     std::cout << "Lookup tables generated and saved." << std::endl;

//     return 0;
// }

int main() {
    // Total LUT size: 256*256*2
    const size_t TOTAL_LUT_SIZE = 256*256*2;
    const size_t E4M3_SIZE = 256;
    const size_t E5M2_SIZE = 256;
    
    // Allocate memory for the combined LUT
    float* combined_lut = new float[TOTAL_LUT_SIZE];
    
    // Initialize LUT to zero
    std::memset(combined_lut, 0, TOTAL_LUT_SIZE * sizeof(float));
    
    // Generate E4M3 FP32 LUT and store in the first 256 entries
    std::cout << "Generating E4M3 FP32 multiplication LUT..." << std::endl;
    generate_fp32_mul_lut(combined_lut, true);  // is_e4m3 = true
    
    // Generate E5M2 FP32 LUT and store in the next 256 entries
    std::cout << "Generating E5M2 FP32 multiplication LUT..." << std::endl;
    generate_fp32_mul_lut(combined_lut + E4M3_SIZE*E4M3_SIZE, false);  // is_e4m3 = false
    
    std::cout << "Saving combined LUT to " << "combined_fp8_mul_lut.bin" << "..." << std::endl;
    save_fp32_lut_to_file(combined_lut, "combined_fp8_mul_lut.bin");
    
    // Clean up
    delete[] combined_lut;
    
    std::cout << "Combined lookup table generated and saved successfully." << std::endl;
    
    return 0;
}