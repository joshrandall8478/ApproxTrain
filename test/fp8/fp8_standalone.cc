#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <bitset>
#include <cstdint>
#include "fp8_conversion.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <cstdlib>
#include <string>
#include <sstream>
#include <iomanip>
#include <tuple>

#include <cuda_fp16.h>

void printFloatBinaryWithSpaces(float value, const std::string& label) {
    union {
        float f;
        uint32_t u;
    } fb;
    fb.f = value;

    std::string bits = std::bitset<32>(fb.u).to_string();
    std::string sign_bit = bits.substr(0, 1);
    std::string exponent_bits = bits.substr(1, 8);
    std::string mantissa_bits = bits.substr(9);

    std::cout << label << " " << sign_bit << " " << exponent_bits << " " << mantissa_bits << std::endl;
}
//main function, generate random float values, convert them to e4m3 and back to float and compare new fp32_e4m3 with fp32_e4m3_old
int main() {
    // Initialize random number generator for normal distribution (mean=0, stddev=1)
    std::mt19937 rng;
    // fix random seed for reproducibility using rng.seed(0)
    rng.seed(0);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    // get size of edge_cases
    int edge_cases_size = edge_cases_e5m2.size();
    #define TESTNUM 100000
    // Initialize matrices with random values
    std::vector<float> matA(TESTNUM+edge_cases_size);
    std::vector<float> matA_e5m2(TESTNUM+edge_cases_size);
    std::vector<float> matA_e5m2_rtz(TESTNUM+edge_cases_size);
    std::vector<float> matA_e4m3(TESTNUM+edge_cases_size);
    std::vector<float> matA_fp16(TESTNUM+edge_cases_size);
    for (int i = 0; i < TESTNUM; ++i) {
        matA[i] = dist(rng);
        matA_e5m2[i] = clip_fp8_e5m2(matA[i],15);
        matA_e5m2_rtz[i] = clip_fp8_e5m2_rtz(matA[i]);
        matA_e4m3[i] = clip_fp8_e4m3(matA[i],7);
        matA_fp16[i] = clip_fp16(matA[i]);
    }
    // append edge cases to the end of the matrices matA
    for (int i = 0; i < edge_cases_size; ++i) {
        matA[TESTNUM+i] = edge_cases_e5m2[i];
        matA_e5m2[TESTNUM+i] = clip_fp8_e5m2(edge_cases_e5m2[i],15);
        matA_e5m2_rtz[TESTNUM+i] = clip_fp8_e5m2_rtz(edge_cases_e5m2[i]);
        matA_e4m3[TESTNUM+i] = clip_fp8_e4m3(edge_cases_e5m2[i],7);
        matA_fp16[TESTNUM+i] = clip_fp16(edge_cases_e5m2[i]);
    }
    // append edge cases e4m3 to the end of the matrices matA
    for (int i = 0; i < edge_cases_size; ++i) {
        matA[TESTNUM+i] = edge_cases_e4m3[i];
        matA_e5m2[TESTNUM+i] = clip_fp8_e5m2(edge_cases_e4m3[i],15);
        matA_e5m2_rtz[TESTNUM+i] = clip_fp8_e5m2_rtz(edge_cases_e4m3[i]);
        matA_e4m3[TESTNUM+i] = clip_fp8_e4m3(edge_cases_e4m3[i],7);
        matA_fp16[TESTNUM+i] = clip_fp16(edge_cases_e4m3[i]);
    }
    // test edge cases here

    // a list to record the difference between matA_e5m2 and matA_e5m2_rtz
    std::vector<float> diff(TESTNUM+edge_cases_size);
    // compare matA_fp32 with matA
    for (int i = 0; i < TESTNUM+edge_cases_size; ++i) {
        // print matA, matA_e4m3_old, matA_e4m3_new
        std::cout << "matA[" << i << "]:          " << matA[i] << std::endl;
        std::cout << "matA_e5m2[" << i << "]:     " << matA_e5m2[i] << std::endl;
        std::cout << "matA_e5m2_rtz[" << i << "]: " << matA_e5m2_rtz[i] << std::endl;
        std::cout << "matA_e4m3[" << i << "]:     " << matA_e4m3[i] << std::endl;
        std::cout << "matA_fp16[" << i << "]:     " << matA_fp16[i] << std::endl;
        // try to calculate the absolute relative difference between matA_e5m2 and matA_e5m2_rtz, if not possible catch the exception
        try {
            diff[i] = std::abs((matA_e5m2[i] - matA_e5m2_rtz[i]) / matA_e5m2_rtz[i]);
        } catch (std::exception& e) {
            std::cerr << "Exception:                    " << e.what() << std::endl;
        }
        // print the absolute relative difference
        std::cout << "Relative difference:              " << diff[i] << std::endl;

        
        // print binary representation of matA, matA_e4m3_old, matA_e4m3_new
        Float32Bits fb;
        fb.f = matA[i];
        std::cout << "matA[" << i << "]          binary: " << std::bitset<32>(fb.u) << std::endl;
        fb.f = matA_e5m2[i];
        std::cout << "matA_e5m2[" << i << "]     binary: " << std::bitset<32>(fb.u) << std::endl;
        fb.f = matA_e5m2_rtz[i];
        std::cout << "matA_e5m2_rtz[" << i << "] binary: " << std::bitset<32>(fb.u) << std::endl;
        fb.f = matA_e4m3[i];
        std::cout << "matA_e4m3[" << i << "]     binary: " << std::bitset<32>(fb.u) << std::endl;
        fb.f = matA_fp16[i];
        std::cout << "matA_fp16[" << i << "]     binary: " << std::bitset<32>(fb.u) << std::endl;
        // print space
        std::cout << std::endl;
    }
    // get the maximum relative difference and print it
    float max_diff = *std::max_element(diff.begin(), diff.end());
    // print its relavent matA_e5m2 and matA_e5m2_rtz
    std::cout << "Max relative difference: " << max_diff << std::endl;
    std::cout << "Max relative difference matA_e5m2: " << matA_e5m2[std::distance(diff.begin(), std::max_element(diff.begin(), diff.end()))] << std::endl;
    std::cout << "Max relative difference matA_e5m2_rtz: " << matA_e5m2_rtz[std::distance(diff.begin(), std::max_element(diff.begin(), diff.end()))] << std::endl;
    std::cout << "Max relative difference matA_e4m3: " << matA_e4m3[std::distance(diff.begin(), std::max_element(diff.begin(), diff.end()))] << std::endl;
    std::cout << "Max relative difference matA_fp16: " << matA_fp16[std::distance(diff.begin(), std::max_element(diff.begin(), diff.end()))] << std::endl;
    // print their binary representation
    Float32Bits fb;
    fb.f = matA[std::distance(diff.begin(), std::max_element(diff.begin(), diff.end()))];
    std::cout << "Max relative difference matA binary: " << std::bitset<32>(fb.u) << std::endl;
    fb.f = matA_e5m2[std::distance(diff.begin(), std::max_element(diff.begin(), diff.end()))];
    std::cout << "Max relative difference matA_e5m2 binary: " << std::bitset<32>(fb.u) << std::endl;
    fb.f = matA_e5m2_rtz[std::distance(diff.begin(), std::max_element(diff.begin(), diff.end()))];
    std::cout << "Max relative difference matA_e5m2_rtz binary: " << std::bitset<32>(fb.u) << std::endl;
    fb.f = matA_e4m3[std::distance(diff.begin(), std::max_element(diff.begin(), diff.end()))];
    std::cout << "Max relative difference matA_e4m3 binary: " << std::bitset<32>(fb.u) << std::endl;
    fb.f = matA_fp16[std::distance(diff.begin(), std::max_element(diff.begin(), diff.end()))];
    std::cout << "Max relative difference matA_fp16 binary: " << std::bitset<32>(fb.u) << std::endl;

    // get all the relative differences that are greater than 0.0 and their corresponding matA, matA_e5m2 and matA_e5m2_rtz
    std::vector<float> diff_greater_than_zero;
    std::vector<float> matA_diff_greater_than_zero;
    std::vector<float> matA_e5m2_diff_greater_than_zero;
    std::vector<float> matA_e5m2_rtz_diff_greater_than_zero;
    std::vector<float> matA_e4m3_diff_greater_than_zero;
    std::vector<float> matA_fp16_diff_greater_than_zero;
    for (int i = 0; i < TESTNUM+edge_cases_size; ++i) {
        if (diff[i] > 0.0) {
            diff_greater_than_zero.push_back(diff[i]);
            matA_diff_greater_than_zero.push_back(matA[i]);
            matA_e5m2_diff_greater_than_zero.push_back(matA_e5m2[i]);
            matA_e5m2_rtz_diff_greater_than_zero.push_back(matA_e5m2_rtz[i]);
            matA_e4m3_diff_greater_than_zero.push_back(matA_e4m3[i]);
            matA_fp16_diff_greater_than_zero.push_back(matA_fp16[i]);
        }
    }
    // sort the relative differences that are greater than 0.0 and keep their corresponding matA, matA_e5m2 and matA_e5m2_rtz in the same order
    std::vector<float> diff_greater_than_zero_sorted = diff_greater_than_zero;
    std::vector<float> matA_diff_greater_than_zero_sorted = matA_diff_greater_than_zero;
    std::vector<float> matA_e5m2_diff_greater_than_zero_sorted = matA_e5m2_diff_greater_than_zero;
    std::vector<float> matA_e5m2_rtz_diff_greater_than_zero_sorted = matA_e5m2_rtz_diff_greater_than_zero;
    std::vector<float> matA_e4m3_diff_greater_than_zero_sorted = matA_e4m3_diff_greater_than_zero;
    std::vector<float> matA_fp16_diff_greater_than_zero_sorted = matA_fp16_diff_greater_than_zero;

    // Combine the vectors into a vector of tuples
std::vector<std::tuple<float, float, float, float, float,float>> combined;
for (size_t i = 0; i < diff_greater_than_zero.size(); ++i) {
    combined.emplace_back(
        diff_greater_than_zero[i],
        matA_diff_greater_than_zero[i],
        matA_e5m2_diff_greater_than_zero[i],
        matA_e5m2_rtz_diff_greater_than_zero[i],
        matA_e4m3_diff_greater_than_zero[i],
        matA_fp16_diff_greater_than_zero[i]);

}

// Sort the combined vector based on diff_greater_than_zero
std::sort(combined.begin(), combined.end(),
    [](const std::tuple<float, float, float, float, float, float>& a,
       const std::tuple<float, float, float, float, float, float>& b) {
        return std::get<0>(a) < std::get<0>(b);
    });

// Unpack the sorted tuples back into the sorted vectors
for (size_t i = 0; i < combined.size(); ++i) {
    diff_greater_than_zero_sorted[i] = std::get<0>(combined[i]);
    matA_diff_greater_than_zero_sorted[i] = std::get<1>(combined[i]);
    matA_e5m2_diff_greater_than_zero_sorted[i] = std::get<2>(combined[i]);
    matA_e5m2_rtz_diff_greater_than_zero_sorted[i] = std::get<3>(combined[i]);
    matA_e4m3_diff_greater_than_zero_sorted[i] = std::get<4>(combined[i]);
    matA_fp16_diff_greater_than_zero_sorted[i] = std::get<5>(combined[i]);
}
    // print separator
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

// print the sorted relative differences that are greater than 0.0 and their corresponding matA, matA_e5m2 and matA_e5m2_rtz
for (int i = 0; i < diff_greater_than_zero_sorted.size(); ++i) {
    std::cout << "Relative difference:              " << diff_greater_than_zero_sorted[i] << std::endl;
    std::cout << "matA[" << i << "]:            " << matA_diff_greater_than_zero_sorted[i] << std::endl;
    std::cout << "matA_e5m2[" << i << "]:       " << matA_e5m2_diff_greater_than_zero_sorted[i] << std::endl;
    std::cout << "matA_e5m2_rtz[" << i << "]:   " << matA_e5m2_rtz_diff_greater_than_zero_sorted[i] << std::endl;
    std::cout << "matA_e4m3[" << i << "]:       " << matA_e4m3_diff_greater_than_zero_sorted[i] << std::endl;
    std::cout << "matA_fp16[" << i << "]:       " << matA_fp16_diff_greater_than_zero_sorted[i] << std::endl;
    // print binary representation of matA, matA_e4m3_old, matA_e4m3_new
    std::string label = "matA[" + std::to_string(i) + "]            binary: ";
    printFloatBinaryWithSpaces(matA_diff_greater_than_zero_sorted[i], label);
    label =             "matA_e5m2[" + std::to_string(i) + "]       binary: ";
    printFloatBinaryWithSpaces(matA_e5m2_diff_greater_than_zero_sorted[i], label);
    label =             "matA_e5m2_rtz[" + std::to_string(i) + "]   binary: ";
    printFloatBinaryWithSpaces(matA_e5m2_rtz_diff_greater_than_zero_sorted[i], label);
    label =             "matA_e4m3[" + std::to_string(i) + "]       binary: ";
    printFloatBinaryWithSpaces(matA_e4m3_diff_greater_than_zero_sorted[i], label);
    uint8_t e4m3 = fp32_to_fp8_e4m3(matA_diff_greater_than_zero_sorted[i],7);
    std::cout << "e4m3: " << std::bitset<8>(e4m3) << std::endl;
    label =             "matA_fp16[" + std::to_string(i) + "]       binary: ";
    printFloatBinaryWithSpaces(matA_fp16_diff_greater_than_zero_sorted[i], label);
    // // if i == 49510, run clip again but to e5m2 only
    // if (i == 49511) {
    //     uint8_t e5m2 = fp32_to_fp8_e5m2(matA_diff_greater_than_zero_sorted[i],15);
    //     __half b = __float2half(matA_diff_greater_than_zero_sorted[i]);
    //     // print e5m2 in bits
    //     std::cout << "e5m2: " << std::bitset<8>(e5m2) << std::endl;
    //     std::cout << "FP16 bits: " << std::bitset<16>(*reinterpret_cast<uint16_t*>(&b)) << std::endl;

    // }
}
    // print count of relative differences that are greater than 0.0
    std::cout << "Count of relative differences greater than 0.0: " << diff_greater_than_zero.size() << std::endl;
    std::cerr << "Conversion successful." << std::endl;
    return EXIT_SUCCESS;
}