#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>

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



////////////////////////////////////////////////////////////////////////////////
// FP32 to FP8 e4m3 Conversion
////////////////////////////////////////////////////////////////////////////////
uint8_t fp32_to_e4m3(float f) {
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
uint8_t fp32_to_e5m2(float f) {
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
// convert float to e4m3 then convert it back to float (clipping)
float clip_e4m3(float f) {
    return e4m3_to_fp32(fp32_to_e4m3(f));
}
// similar thing for e5m2
float clip_e5m2(float f) {
    return e5m2_to_fp32(fp32_to_e5m2(f));
}

// Dimensions for matrix1. These should be a multiple of BLOCK
constexpr int ROWS1 = 800;
constexpr int COLS1 = 1600;

// Dimensions for matrix2. These should be a multiple of BLOCK
constexpr int ROWS2 = 1600;
constexpr int COLS2 = 800;


/* Function to perform matrix multiplication */
void matMul(std::vector<float> &matC, const std::vector<float> &matA, const std::vector<float> &matB,
           int rows1, int cols1, int cols2) {
    for(int row = 0; row < rows1; ++row){
        for(int col = 0; col < cols2; ++col){
            float prod = 0.0f;
            for(int k = 0; k < cols1; ++k){
                prod += matA[row * cols1 + k] * matB[k * cols2 + col];
            }
            matC[row * cols2 + col] = prod;
        }
    }
}
/*
matMul same as before, but with e4m3 clipping
*/
void matMul_e4m3(std::vector<float> &matC, const std::vector<float> &matA, const std::vector<float> &matB,
           int rows1, int cols1, int cols2) {
    for(int row = 0; row < rows1; ++row){
        for(int col = 0; col < cols2; ++col){
            float prod = 0.0f;
            for(int k = 0; k < cols1; ++k){
                prod += clip_e4m3(matA[row * cols1 + k]) * clip_e4m3(matB[k * cols2 + col]);
            }
            matC[row * cols2 + col] = prod;
        }
    }
}
/*
same as previous function, but with e5m2 clipping
*/
void matMul_e5m2(std::vector<float> &matC, const std::vector<float> &matA, const std::vector<float> &matB,
           int rows1, int cols1, int cols2) {
    for(int row = 0; row < rows1; ++row){
        for(int col = 0; col < cols2; ++col){
            float prod = 0.0f;
            for(int k = 0; k < cols1; ++k){
                prod += clip_e5m2(matA[row * cols1 + k]) * clip_e5m2(matB[k * cols2 + col]);
            }
            matC[row * cols2 + col] = prod;
        }
    }
}

int main(){
    // Verify matrix dimensions
    if(COLS1 != ROWS2){
        std::cerr << "Matrix dimensions are invalid for matrix multiplication." << std::endl;
        return EXIT_FAILURE;
    }

    // Initialize random number generator for normal distribution (mean=0, stddev=1)
    
    
    std::mt19937 rng;
    // fix random seed for reproducibility using rng.seed(0)
    rng.seed(0);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Initialize matrices with random values
    std::vector<float> matA(ROWS1 * COLS1);
    std::vector<float> matB(ROWS2 * COLS2);
    std::vector<float> matC(ROWS1 * COLS2, 0.0f); // Initialize matC with zeros

    for(int i = 0; i < ROWS1; ++i){
        for(int j = 0; j < COLS1; ++j){
            matA[i * COLS1 + j] = dist(rng);
        }
    }

    for(int i = 0; i < ROWS2; ++i){
        for(int j = 0; j < COLS2; ++j){
            matB[i * COLS2 + j] = dist(rng);
        }
    }
    // save matA and matB to file
    std::ofstream ofs("matA.bin", std::ios::binary);
    if (!ofs) {
        std::cerr << "Error opening file for writing: " << "matA.bin" << std::endl;
        return EXIT_FAILURE;
    }
    ofs.write(reinterpret_cast<const char*>(matA.data()), ROWS1 * COLS1 * sizeof(float));
    ofs.close();
    if (!ofs.good()) {
        std::cerr << "Error occurred while writing to file: " << "matA.bin" << std::endl;
        return EXIT_FAILURE;
    }
    ofs.open("matB.bin", std::ios::binary);
    if (!ofs) {
        std::cerr << "Error opening file for writing: " << "matB.bin" << std::endl;
        return EXIT_FAILURE;
    }
    ofs.write(reinterpret_cast<const char*>(matB.data()), ROWS2 * COLS2 * sizeof(float));
    ofs.close();
    if (!ofs.good()) {
        std::cerr << "Error occurred while writing to file: " << "matB.bin" << std::endl;
        return EXIT_FAILURE;
    }

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Perform matrix multiplication
    matMul(matC, matA, matB, ROWS1, COLS1, COLS2);

    // Stop timing
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = stop - start;
    // save matC to file
    ofs.open("matC.bin", std::ios::binary);
    if (!ofs) {
        std::cerr << "Error opening file for writing: " << "matC.bin" << std::endl;
        return EXIT_FAILURE;
    }
    ofs.write(reinterpret_cast<const char*>(matC.data()), ROWS1 * COLS2 * sizeof(float));
    ofs.close();
    if (!ofs.good()) {
        std::cerr << "Error occurred while writing to file: " << "matC.bin" << std::endl;
        return EXIT_FAILURE;
    }

    // Optionally, print the result matrix (commented out to avoid large output)
    /*
    std::cout << "Result Matrix C:" << std::endl;
    for(int i = 0; i < ROWS1; ++i){
        for(int j = 0; j < COLS2; ++j){
            std::cout << matC[i * COLS2 + j] << " ";
        }
        std::cout << std::endl;
    }
    */

    // Print the elapsed time
    std::cerr << "Elapsed time for matrix multiplication on CPU: " << elapsed.count() << " seconds." << std::endl;

    // Start timing
    start = std::chrono::high_resolution_clock::now();
    // Perform matrix multiplication with e4m3 clipping
    matMul_e4m3(matC, matA, matB, ROWS1, COLS1, COLS2);
    // Stop timing
    stop = std::chrono::high_resolution_clock::now();
    elapsed = stop - start;
    // Print the elapsed time
    std::cerr << "Elapsed time for matrix multiplication on CPU with e4m3 clipping: " << elapsed.count() << " seconds." << std::endl;
    // save matC to file
    ofs.open("matC_e4m3.bin", std::ios::binary);
    if (!ofs) {
        std::cerr << "Error opening file for writing: " << "matC_e4m3.bin" << std::endl;
        return EXIT_FAILURE;
    }
    ofs.write(reinterpret_cast<const char*>(matC.data()), ROWS1 * COLS2 * sizeof(float));
    ofs.close();
    if (!ofs.good()) {
        std::cerr << "Error occurred while writing to file: " << "matC_e4m3.bin" << std::endl;
        return EXIT_FAILURE;
    }

    // Start timing
    start = std::chrono::high_resolution_clock::now();
    // Perform matrix multiplication with e5m2 clipping
    matMul_e5m2(matC, matA, matB, ROWS1, COLS1, COLS2);
    // Stop timing
    stop = std::chrono::high_resolution_clock::now();
    elapsed = stop - start;
    // Print the elapsed time
    std::cerr << "Elapsed time for matrix multiplication on CPU with e5m2 clipping: " << elapsed.count() << " seconds." << std::endl;
    // save matC to file
    ofs.open("matC_e5m2.bin", std::ios::binary);
    if (!ofs) {
        std::cerr << "Error opening file for writing: " << "matC_e5m2.bin" << std::endl;
        return EXIT_FAILURE;
    }
    ofs.write(reinterpret_cast<const char*>(matC.data()), ROWS1 * COLS2 * sizeof(float));
    ofs.close();
    if (!ofs.good()) {
        std::cerr << "Error occurred while writing to file: " << "matC_e5m2.bin" << std::endl;
        return EXIT_FAILURE;
    }
    

    return EXIT_SUCCESS;
}
