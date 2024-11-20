#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <error.cuh>
#include <fp8_conversion.cuh>
//Dimensions for matrix1. These should be a multiple of BLOCK
#define ROWS1 800
#define COLS1 1600

//DImensions for matrix2. These should be a multiple of BLOCK
#define ROWS2 1600
#define COLS2 800

// define TILE_DIM
#define TILE_DIM 16
// define BLOCK
#define BLOCK 16
template <typename T>
__global__ void gemm(size_t m, size_t n, size_t k,
    const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc)
{
    T value(0);

    int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    int Col = blockIdx.x*TILE_DIM + threadIdx.x;

    __shared__ T As[TILE_DIM][TILE_DIM];
    __shared__ T Bs[TILE_DIM][TILE_DIM];

    for (int i = 0; i < (TILE_DIM + k - 1)/TILE_DIM; ++i) {

        if (i*TILE_DIM + threadIdx.x < k && Row < m){
            As[threadIdx.y][threadIdx.x] = a[Row*lda + i*TILE_DIM + threadIdx.x];
        }
        else{
            As[threadIdx.y][threadIdx.x] = T(0);
        }

        if (i*TILE_DIM + threadIdx.y < k && Col < n){
            Bs[threadIdx.y][threadIdx.x] = b[(i*TILE_DIM + threadIdx.y)*ldb + Col];
        }
        else{
            Bs[threadIdx.y][threadIdx.x] = T(0);
        }

        __syncthreads();

        for (int n = 0; n < TILE_DIM; ++n){
            value += As[threadIdx.y][n]*Bs[n][threadIdx.x];
        }

        __syncthreads();
    }

    if (Row < m && Col < n) {
        c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
    }
}  
template <typename T>
__global__ void gemm_e4m3(size_t m, size_t n, size_t k,
    const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc, cudaTextureObject_t mant_lut)
{
    T value(0);

    int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    int Col = blockIdx.x*TILE_DIM + threadIdx.x;

    __shared__ T As[TILE_DIM][TILE_DIM];
    __shared__ T Bs[TILE_DIM][TILE_DIM];

    for (int i = 0; i < (TILE_DIM + k - 1)/TILE_DIM; ++i) {

        if (i*TILE_DIM + threadIdx.x < k && Row < m){
            As[threadIdx.y][threadIdx.x] = a[Row*lda + i*TILE_DIM + threadIdx.x];
        }
        else{
            As[threadIdx.y][threadIdx.x] = T(0);
        }

        if (i*TILE_DIM + threadIdx.y < k && Col < n){
            Bs[threadIdx.y][threadIdx.x] = b[(i*TILE_DIM + threadIdx.y)*ldb + Col];
        }
        else{
            Bs[threadIdx.y][threadIdx.x] = T(0);
        }

        __syncthreads();

        for (int j = 0; j < TILE_DIM; ++j) {
            uint8_t a_key = fp32_to_e4m3(As[threadIdx.y][j]);
            uint8_t b_key = fp32_to_e4m3(Bs[j][threadIdx.x]);

            // Compute the index into the LUT
            uint32_t index = (a_key << 8) | b_key;  // Concatenate a_key and b_key

            // Fetch the multiplication result from the LUT
            float mul_result = tex1Dfetch<float>(mant_lut, index);

            // Accumulate the result
            value += mul_result;
        }

        __syncthreads();
    }

    if (Row < m && Col < n) {
        c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
    }
}
template __global__ void gemm<float>(size_t m, size_t n, size_t k,
    const float *a, size_t lda, const float *b, size_t ldb,
   float *c, size_t ldc);

    
template __global__ void gemm_e4m3<float>(size_t m, size_t n, size_t k,
    const float *a, size_t lda, const float *b, size_t ldb,
   float *c, size_t ldc, cudaTextureObject_t mant_lut);



template <typename T>
__global__ void gemm_e5m2(size_t m, size_t n, size_t k,
    const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc, cudaTextureObject_t mant_lut)
{
    T value(0);

    int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    int Col = blockIdx.x*TILE_DIM + threadIdx.x;

    __shared__ T As[TILE_DIM][TILE_DIM];
    __shared__ T Bs[TILE_DIM][TILE_DIM];

    for (int i = 0; i < (TILE_DIM + k - 1)/TILE_DIM; ++i) {

        if (i*TILE_DIM + threadIdx.x < k && Row < m){
            As[threadIdx.y][threadIdx.x] = a[Row*lda + i*TILE_DIM + threadIdx.x];
        }
        else{
            As[threadIdx.y][threadIdx.x] = T(0);
        }

        if (i*TILE_DIM + threadIdx.y < k && Col < n){
            Bs[threadIdx.y][threadIdx.x] = b[(i*TILE_DIM + threadIdx.y)*ldb + Col];
        }
        else{
            Bs[threadIdx.y][threadIdx.x] = T(0);
        }

        __syncthreads();

        for (int j = 0; j < TILE_DIM; ++j) {
            uint8_t a_key = fp32_to_e5m2(As[threadIdx.y][j]);
            uint8_t b_key = fp32_to_e5m2(Bs[j][threadIdx.x]);

            // Compute the index into the LUT
            uint32_t index = ((a_key << 8) | b_key)+ 256*256;  // Concatenate a_key and b_key

            // Fetch the multiplication result from the LUT
            float mul_result = tex1Dfetch<float>(mant_lut, index);

            // Accumulate the result
            value += mul_result;
        }

        __syncthreads();
    }

    if (Row < m && Col < n) {
        c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
    }
}

template __global__ void gemm_e5m2<float>(size_t m, size_t n, size_t k,
    const float *a, size_t lda, const float *b, size_t ldb,
   float *c, size_t ldc, cudaTextureObject_t mant_lut);

// main function
int main(){

    //check whether dimensions are valid for matrix mutiplication
    if(COLS1!=ROWS2){
        printf("Matrix dimensions are invalid for matrix multiplication\n");
        exit(1);
    }
    //check whether the requirements for the current version of the program is met
    if(COLS1%BLOCK!=0 || COLS2%BLOCK!=0 || ROWS1%BLOCK!=0 || ROWS2%BLOCK!=0){
        fprintf(stderr, "This program need the COLS1 COLS2 ROWS1 and ROWS2 to be multiples of BLOCK\n");
        exit(1);
    }

    //Initialize arrays in RAM
    float *matA = (float *)malloc(sizeof(float)*ROWS1*COLS1);
    float *matB = (float *)malloc(sizeof(float)*ROWS2*COLS2);
    float *matC = (float *)malloc(sizeof(float)*ROWS1*COLS2);
    // array for matC_e4m3
    float *matC_e4m3 = (float *)malloc(sizeof(float)*ROWS1*COLS2);
    // array for matC_e4m3_ref
    float *matC_e4m3_ref = (float *)malloc(sizeof(float)*ROWS1*COLS2);
    // array for matC_e5m2
    float *matC_e5m2 = (float *)malloc(sizeof(float)*ROWS1*COLS2);
    // array for matC_e5m2_ref
    float *matC_e5m2_ref = (float *)malloc(sizeof(float)*ROWS1*COLS2);
    // array for matC_ref
    float *matC_ref = (float *)malloc(sizeof(float)*ROWS1*COLS2);
    //check if out of memory.
    if(matA==NULL || matB==NULL || matC==NULL){
        perror("Memory out");
        exit(EXIT_FAILURE);
    }

    //generate some values for matrixA from reading matA.bin
    FILE *f = fopen("test/fp8/matA.bin", "rb");
    if (f == NULL) {
        printf("matA.bin Error opening file!\n");
        exit(1);
    }
    fread(matA, sizeof(float), ROWS1*COLS1, f);
    fclose(f);
    // generate some values for matrixB from reading matB.bin
    f = fopen("test/fp8/matB.bin", "rb");
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }
    fread(matB, sizeof(float), ROWS2*COLS2, f);
    fclose(f);
    // read the expected output from matC_e4m3.bin
    f = fopen("test/fp8/matC_e4m3.bin", "rb");
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }
    fread(matC_e4m3_ref, sizeof(float), ROWS1*COLS2, f);
    fclose(f);
    // read the expected output from matC_e5m2.bin
    f = fopen("test/fp8/matC_e5m2.bin", "rb");
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }
    fread(matC_e5m2_ref, sizeof(float), ROWS1*COLS2, f);
    fclose(f);
    // read the expected output from matC.bin
    f = fopen("test/fp8/matC.bin", "rb");
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }
    fread(matC_ref, sizeof(float), ROWS1*COLS2, f);
    fclose(f);
    /********************************** CUDA stuff starts here *******************************/

	//start meauring time
	cudaEvent_t start,stop;
	float elapsedtime;
	cudaEventCreate(&start);
	cudaEventRecord(start,0);

	//pointers for memory allocation in cudaa
	float *matA_cuda;
	float *matB_cuda;
	float *matC_cuda;
    // pointers for memory allocation in cuda of matC_e4m3
    float *matC_e4m3_cuda;
    // pointers for memory allocation in cuda of matC_e5m2
    float *matC_e5m2_cuda;

	//allocate memory in cuda
	cudaMalloc((void **)&matA_cuda,sizeof(float)*ROWS1*COLS1); checkCudaError();
	cudaMalloc((void **)&matB_cuda,sizeof(float)*ROWS2*COLS2); checkCudaError();
	cudaMalloc((void **)&matC_cuda,sizeof(float)*ROWS1*COLS2); checkCudaError();
    // allocate memory in cuda of matC_e4m3
    cudaMalloc((void **)&matC_e4m3_cuda,sizeof(float)*ROWS1*COLS2); checkCudaError();
    // allocate memory in cuda of matC_e5m2
    cudaMalloc((void **)&matC_e5m2_cuda,sizeof(float)*ROWS1*COLS2); checkCudaError();

	//copy memory from ram to cuda
	cudaMemcpy(matA_cuda,matA,sizeof(float)*ROWS1*COLS1,cudaMemcpyHostToDevice); checkCudaError();
	cudaMemcpy(matB_cuda,matB,sizeof(float)*ROWS2*COLS2,cudaMemcpyHostToDevice); checkCudaError();

	//multiply the matrices
	dim3 threadsPerBlock(BLOCK,BLOCK);
	dim3 numBlocks(ceil(COLS2/(float)BLOCK),ceil(ROWS1/(float)BLOCK));

	//start measuring time for cuda kernel only
	cudaEvent_t startkernel,stopkernel;
	float elapsedtimekernel;
	cudaEventCreate(&startkernel);
	cudaEventRecord(startkernel,0);


	gemm<<<numBlocks,threadsPerBlock>>>(ROWS1,COLS2,COLS1,matA_cuda, COLS1,matB_cuda, COLS2,matC_cuda,COLS2);
	cudaDeviceSynchronize(); checkCudaError();

	//end measuring time for cuda kernel
	cudaEventCreate(&stopkernel);
	cudaEventRecord(stopkernel,0);
	cudaEventSynchronize(stopkernel);
	cudaEventElapsedTime(&elapsedtimekernel,startkernel,stopkernel);

	//copy the answer back from cuda ro ram
	cudaMemcpy(matC,matC_cuda,sizeof(float)*ROWS1*COLS2,cudaMemcpyDeviceToHost); checkCudaError();



	//end measuring time
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedtime,start,stop);
    //print the time spent to stderr
	fprintf(stderr,"Time spent for CUDA kernel is %1.5f seconds\n",elapsedtimekernel/(float)1000);
	fprintf(stderr,"Time spent for operation on CUDA(Including memory allocation and copying) is %1.5f seconds\n",elapsedtime/(float)1000);
    
    //start of e4m3
    // create cuda texture object 
    cudaTextureObject_t mant_mul_lut_;
    // create cuda vector for mant_mul_lut_fp32_
    float *mant_mul_lut_cuda_fp32_;
    // Open mant mul file
    std::ifstream file("test/fp8/combined_fp8_mul_lut.bin", std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: failed to open combined_fp8_mul_lut file" << std::endl;
        return EXIT_FAILURE;
    }

    // mant_mul_lut_fp32_ float pointer
    std::vector<float> mant_mul_lut_fp32_(256*256*2, 0);

    // read the file
    file.read(
        reinterpret_cast<char*>(mant_mul_lut_fp32_.data()),
        mant_mul_lut_fp32_.size() * sizeof(float)
    );
    // close the file
    file.close();



    // Handle combined FP8 LUT
    cudaMalloc(&mant_mul_lut_cuda_fp32_,
                            mant_mul_lut_fp32_.size() * sizeof(float));
    // check for cuda error
    checkCudaError();
    cudaMemcpy(mant_mul_lut_cuda_fp32_, mant_mul_lut_fp32_.data(),
                            mant_mul_lut_fp32_.size() * sizeof(float),
                            cudaMemcpyHostToDevice);
    // check for cuda error
    checkCudaError();


    // Create texture object for combined FP8 LUT
    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(cudaResourceDesc));
    res_desc.resType = cudaResourceTypeLinear;
    res_desc.res.linear.devPtr = mant_mul_lut_cuda_fp32_;
    res_desc.res.linear.desc.f = cudaChannelFormatKindFloat;
    res_desc.res.linear.desc.x = 32;
    res_desc.res.linear.sizeInBytes =
        mant_mul_lut_fp32_.size() * sizeof(float);

    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(cudaTextureDesc));
    tex_desc.readMode = cudaReadModeElementType;

    cudaCreateTextureObject(&mant_mul_lut_,
                                        &res_desc,
                                        &tex_desc, nullptr);   

    
    // start measuring time for cuda kernel only
    cudaEventCreate(&startkernel);
    cudaEventRecord(startkernel,0);

    // call the gemm_e4m3 kernel
    gemm_e4m3<<<numBlocks,threadsPerBlock>>>(ROWS1,COLS2,COLS1,matA_cuda, COLS1,matB_cuda, COLS2,matC_e4m3_cuda,COLS2, mant_mul_lut_);
    // check for cuda error
    cudaDeviceSynchronize(); checkCudaError();
    // end measuring time for cuda kernel
    cudaEventCreate(&stopkernel);
    cudaEventRecord(stopkernel,0);
    cudaEventSynchronize(stopkernel);
    cudaEventElapsedTime(&elapsedtimekernel,startkernel,stopkernel);
    // copy the answer back from cuda ro ram
    cudaMemcpy(matC_e4m3,matC_e4m3_cuda,sizeof(float)*ROWS1*COLS2,cudaMemcpyDeviceToHost); checkCudaError();

    // print the time spent to stderr
    fprintf(stderr,"Time spent for CUDA kernel is %1.5f seconds\n",elapsedtimekernel/(float)1000);
    // end of e4m3

    


    //start of e5m2
    // start measuring time for cuda kernel only
    cudaEventCreate(&startkernel);
    cudaEventRecord(startkernel,0);
    // call the gemm_e5m2 kernel
    gemm_e5m2<<<numBlocks,threadsPerBlock>>>(ROWS1,COLS2,COLS1,matA_cuda, COLS1,matB_cuda, COLS2,matC_e5m2_cuda,COLS2, mant_mul_lut_);

    // check for cuda error
    cudaDeviceSynchronize(); checkCudaError();
    // end measuring time for cuda kernel
    cudaEventCreate(&stopkernel);
    cudaEventRecord(stopkernel,0);
    cudaEventSynchronize(stopkernel);
    cudaEventElapsedTime(&elapsedtimekernel,startkernel,stopkernel);
    // copy the answer back from cuda ro ram
    cudaMemcpy(matC_e5m2,matC_e5m2_cuda,sizeof(float)*ROWS1*COLS2,cudaMemcpyDeviceToHost); checkCudaError();
    // free all memory
    cudaDestroyTextureObject(mant_mul_lut_);
    cudaFree(mant_mul_lut_cuda_fp32_);
    // free the cuda memory
    cudaFree(matA_cuda); checkCudaError();
    cudaFree(matB_cuda); checkCudaError();
    cudaFree(matC_cuda); checkCudaError();
    cudaFree(matC_e4m3_cuda); checkCudaError();
    cudaFree(matC_e5m2_cuda); checkCudaError();
    
    // print the time spent to stderr
    fprintf(stderr,"Time spent for CUDA kernel is %1.5f seconds\n",elapsedtimekernel/(float)1000);
    // end of e5m2

    // // compare matC with matC_ref
    // for (int i = 0; i < ROWS1*COLS2; i++) {
    //     if (fabs(matC[i] - matC_ref[i]) > 1e-5) {
    //         fprintf(stderr, "matC[%d] = %f, matC_ref[%d] = %f\n", i, matC[i], i, matC_ref[i]);
    //         fprintf(stderr, "matC and matC_ref are not equal\n");
    //         exit(EXIT_FAILURE);
    //     }
    // }
    // compare matC_e4m3 with matC_e4m3 of cuda
    for (int i = 0; i < ROWS1*COLS2; i++) {
        if (fabs(matC_e4m3[i] - matC_e4m3_ref[i]) > 1e-5) {
            fprintf(stderr, "matC_e4m3[%d] = %f, matC_e4m3_cuda[%d] = %f\n", i, matC_e4m3[i], i, matC_e4m3_ref[i]);
            fprintf(stderr, "matC_e4m3 and matC_e4m3_cuda are not equal\n");
            exit(EXIT_FAILURE);
        }
    }
    // compare matC_e5m2 with matC_e5m2 of cuda
    for (int i = 0; i < ROWS1*COLS2; i++) {
        if (fabs(matC_e5m2[i] - matC_e5m2_ref[i]) > 1e-5) {
            fprintf(stderr, "matC_e5m2[%d] = %f, matC_e5m2_cuda[%d] = %f\n", i, matC_e5m2[i], i, matC_e5m2_ref[i]);
            fprintf(stderr, "matC_e5m2 and matC_e5m2_cuda are not equal\n");
            exit(EXIT_FAILURE);
        }
    }

    

	/********************** CUDA stuff ends here ********************************/

    return EXIT_SUCCESS;
}