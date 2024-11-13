#include <iostream>
#include <bitset>
#include <cstring>  // Add this for std::memcpy
#include <cuda_runtime.h>
// Kernel to perform addition with default rounding mode (nearest even)
__global__ void addDefaultRounding(const float* a, const float* b, float* c, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];  // Default rounding (nearest even)
    }
}

// Kernel to perform addition with rounding toward zero
__global__ void addRoundingTowardZero(const float* a, const float* b, float* c, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        c[idx] = __fadd_rz(a[idx], b[idx]);  // Rounding toward zero
    }
}

// Function to print an array
void printArray(const char* label, const float* arr, int size) {
    std::cout << label << ": ";
    for (int i = 0; i < size; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

// Function to print a float in 32-bit binary format
void printBinary32(const char* label, const float* arr, int size) {
    std::cout << label << ": ";
    for (int i = 0; i < size; ++i) {
        // Treat the float as an int32_t to access the raw bits
        int32_t intRep;
        std::memcpy(&intRep, &arr[i], sizeof(intRep));  // Copy float bits to int
        std::cout << std::bitset<32>(intRep) << " ";
    }
    std::cout << std::endl;
}


float binaryStringToFloat(const std::string& binaryString) {
    // Check if the binary string is exactly 32 bits
    if (binaryString.size() != 32) {
        throw std::invalid_argument("Binary string must be exactly 32 bits long.");
    }

    // Convert the binary string to a 32-bit unsigned integer
    uint32_t intRep = std::bitset<32>(binaryString).to_ulong();

    // Convert the integer representation to a float
    float result;
    std::memcpy(&result, &intRep, sizeof(result));
    return result;
}


int main() {
    const int size = 5;
    /*
    strin_f1
    strin_f2
    are two floats that trigger different behavior of round towards zero and round towards nearest
    this test is for the sake of proper functionality of rounding in CUDA/GPU
    */
    std::string strin_f1 = "01000011001011110000000000001111";
    std::string strin_f2 = "01000100001011110000000000001111";
    float a1 = binaryStringToFloat(strin_f1);
    float b1 = binaryStringToFloat(strin_f2);
    float h_a[size] = {a1, -1.25f, 1.75f, -1.75f, 0.5f};  // Values near halfway points
    float h_b[size] = {b1, -0.75f, 0.25f, -0.25f, 1.5f};  // Values that will trigger rounding
    float h_c_default[size], h_c_rz[size];

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size * sizeof(float));
    cudaMalloc((void**)&d_b, size * sizeof(float));
    cudaMalloc((void**)&d_c, size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_a, h_a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size * sizeof(float), cudaMemcpyHostToDevice);

    // Set up execution configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Run kernel with default rounding
    addDefaultRounding<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, size);
    cudaMemcpy(h_c_default, d_c, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Run kernel with rounding toward zero
    addRoundingTowardZero<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, size);
    cudaMemcpy(h_c_rz, d_c, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the results
    printArray("A", h_a, size);
    printArray("B", h_b, size);
    printArray("Result with default rounding", h_c_default, size);
    printArray("Result with rounding toward zero", h_c_rz, size);

    // Print the inputs and results in binary format
    printBinary32("A in binary", h_a, size);
    printBinary32("B in binary", h_b, size);
    printBinary32("Result with default rounding in binary", h_c_default, size);
    printBinary32("Result with rounding toward zero in binary", h_c_rz, size);
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
