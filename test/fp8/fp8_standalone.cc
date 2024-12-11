#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <bitset>
#include <cstdint>
#include "fp8_conversion.h"
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
    std::vector<float> matA_e5m2(TESTNUM);
    for (int i = 0; i < TESTNUM; ++i) {
        matA[i] = dist(rng);
        matA_e5m2[i] = clip_fp8_e5m2(matA[i],24);
    }
    // compare matA_fp32 with matA
    for (int i = 0; i < TESTNUM; ++i) {
        // print matA, matA_e4m3_old, matA_e4m3_new
        std::cout << "matA[" << i << "]: " << matA[i] << std::endl;
        std::cout << "matA_e5m2[" << i << "]: " << matA_e5m2[i] << std::endl;
        // print binary representation of matA, matA_e4m3_old, matA_e4m3_new
        Float32Bits fb;
        fb.f = matA[i];
        std::cout << "matA[" << i << "] binary: " << std::bitset<32>(fb.u) << std::endl;
        fb.f = matA_e5m2[i];
        std::cout << "matA_e4m3_old[" << i << "] binary: " << std::bitset<32>(fb.u) << std::endl;
        // print space
        std::cout << std::endl;
    }
    std::cerr << "Conversion successful." << std::endl;
    return EXIT_SUCCESS;
}