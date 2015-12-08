
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>

// includes, kernels
#include <tr_kernel.cu>
#include <cublas.h>
////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char** argv);
void randomInit(float*, int);
void printDiff(float*, float*, int, int);

extern "C"
void computeGold(float*, const float*, unsigned int, unsigned int, unsigned int);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv)
{
    runTest(argc, argv);

    CUT_EXIT(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest(int argc, char** argv)
{
    CUT_DEVICE_INIT(argc, argv);


    float result[1024];
    cublasStatus status;
    status = cublasInit();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! CUBLAS initialization error\n");
	exit (1);
    }
    // set seed for rand()
    srand(2006);

    // allocate host memory for matrices A and B
    unsigned int size_A = WA * HA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*) malloc(mem_size_A);

    // initialize host memory
    randomInit(h_A, size_A);

    // allocate device memory
    float* d_A;
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_A, mem_size_A));

    // copy host memory to device
    CUDA_SAFE_CALL(cudaMemcpy(d_A, h_A, mem_size_A,
                              cudaMemcpyHostToDevice) );

    // allocate device memory for result
    unsigned int size_C = WC * HC;
    unsigned int mem_size_C = sizeof(float) * size_C;

    // allocate host memory for the result
    float* h_C = (float*) malloc(mem_size_C);

    // create and start timer
    unsigned int timer = 0;

    // compute reference solution
    float* reference = (float*) malloc(mem_size_C);
    computeGold(reference, h_A, HA, WA, WB);
    CUTBoolean res;

    printf("finish cpu\n");
    int pid = 0;

    {
        free(h_C);
        h_C = (float*) malloc(mem_size_C);
        float* d_C;
        CUDA_SAFE_CALL(cudaMalloc((void**) &d_C, mem_size_C));
		// setup execution parameters
		dim3 threads(16, 16);
		dim3 grid(WC / threads.x, HC / threads.y);

		CUT_SAFE_CALL(cutCreateTimer(&timer));
		cudaThreadSynchronize();
		CUT_SAFE_CALL(cutStartTimer(timer));
		// execute the kernel
		transpose_naive<<< grid, threads >>>(d_A, d_C, WA);
		// stop and destroy timer
		cudaThreadSynchronize();
		CUT_SAFE_CALL(cutStopTimer(timer));

		// check if kernel execution generated and error
		CUT_CHECK_ERROR("Kernel execution failed");

		// copy result from device to host
		CUDA_SAFE_CALL(cudaMemcpy(h_C, d_C, mem_size_C,
								  cudaMemcpyDeviceToHost) );
	    printf("finish gpu\n");

		result[pid++] = cutGetTimerValue(timer);
		printf("transpose_naive Processing time: %f (ms), %f Gflops \n", cutGetTimerValue(timer), 2000.0*4*MW*MW/cutGetTimerValue(timer)/1024/1024/1024);
		CUT_SAFE_CALL(cutDeleteTimer(timer));
	    CUDA_SAFE_CALL(cudaFree(d_C));
    }
    // check result
    res = cutCompareL2fe(reference, h_C, size_C, 1e-6f);
    printf("Test %s \n", (1 == res) ? "PASSED" : "FAILED");

	{
		free(h_C);
		h_C = (float*) malloc(mem_size_C);
		float* d_C;
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_C, mem_size_C));
   		// setup execution parameters
   		dim3 threads(32, 1);
   		dim3 grid(WC / threads.x, HC / threads.y /1);

   		CUT_SAFE_CALL(cutCreateTimer(&timer));
   		cudaThreadSynchronize();
   		CUT_SAFE_CALL(cutStartTimer(timer));
   		// execute the kernel
   		transpose_coalesced<<< grid, threads >>>(d_A, d_C, WA);
   		// stop and destroy timer
   		cudaThreadSynchronize();
   		CUT_SAFE_CALL(cutStopTimer(timer));

   		// check if kernel execution generated and error
   		CUT_CHECK_ERROR("Kernel execution failed");

   		// copy result from device to host
   		CUDA_SAFE_CALL(cudaMemcpy(h_C, d_C, mem_size_C,
   								  cudaMemcpyDeviceToHost) );

   		result[pid++] = cutGetTimerValue(timer);
   		printf("transpose_coalesced Processing time: %f (ms), %f Gflops \n", cutGetTimerValue(timer), 2000.0*4*MW*MW/cutGetTimerValue(timer)/1024/1024/1024);
   		CUT_SAFE_CALL(cutDeleteTimer(timer));
   	    CUDA_SAFE_CALL(cudaFree(d_C));
	}
	// check result
	res = cutCompareL2fe(reference, h_C, size_C, 1e-6f);
	printf("Test %s \n", (1 == res) ? "PASSED" : "FAILED");


	{
		free(h_C);
		h_C = (float*) malloc(mem_size_C);
		float* d_C;
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_C, mem_size_C));
   		// setup execution parameters
   		dim3 threads(32, 4);
   		dim3 grid(WC / threads.x, HC / threads.y /8);

   		CUT_SAFE_CALL(cutCreateTimer(&timer));
   		cudaThreadSynchronize();
   		CUT_SAFE_CALL(cutStartTimer(timer));
   		// execute the kernel
   		transpose_opt<<< grid, threads >>>(d_A, d_C, WA);
   		// stop and destroy timer
   		cudaThreadSynchronize();
   		CUT_SAFE_CALL(cutStopTimer(timer));

   		// check if kernel execution generated and error
   		CUT_CHECK_ERROR("Kernel execution failed");

   		// copy result from device to host
   		CUDA_SAFE_CALL(cudaMemcpy(h_C, d_C, mem_size_C,
   								  cudaMemcpyDeviceToHost) );

   		result[pid++] = cutGetTimerValue(timer);
   		printf("transpose_opt Processing time: %f (ms), %f Gflops \n", cutGetTimerValue(timer), 2000.0*4*MW*MW/cutGetTimerValue(timer)/1024/1024/1024);
   		CUT_SAFE_CALL(cutDeleteTimer(timer));
   	    CUDA_SAFE_CALL(cudaFree(d_C));
	}
	// check result
	res = cutCompareL2fe(reference, h_C, size_C, 1e-6f);
	printf("Test %s \n", (1 == res) ? "PASSED" : "FAILED");


	printf("\n");
    // clean up memory
    free(h_A);
    free(h_C);
    free(reference);
    CUDA_SAFE_CALL(cudaFree(d_A));
    status = cublasShutdown();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! shutdown error\n");
    }
//    CUDA_SAFE_CALL(cudaFree(d_C));
}

// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

void printDiff(float *data1, float *data2, int width, int height)
{
  int i,j,k;
  int error_count=0;
  for (j=0; j<height; j++) {
    for (i=0; i<width; i++) {
      k = j*width+i;
      if (data1[k] != data2[k]) {
         printf("diff(%d,%d) CPU=%4.4f, GPU=%4.4f n", i,j, data1[k], data2[k]);
         error_count++;
      }
    }
  }
  printf(" nTotal Errors = %d n", error_count);
}

