// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>

// includes, kernels
#include <imregionmax_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char** argv);
void randomInit(float*, int);
void printDiff(float*, float*, int, int);

extern "C" void computeGold(float*, const float*, unsigned int, unsigned int);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
	runTest(argc, argv);

	CUT_EXIT(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char** argv) {
	CUT_DEVICE_INIT(argc, argv);

	// set seed for rand()
	srand(2006);

	// allocate host memory for matrices A
	unsigned int size_A = WIDTH_A * HEIGHT_A;
	unsigned int mem_size_A = sizeof(float) * size_A;
	float* h_A = (float*) malloc(mem_size_A);

	// initialize host memory
	randomInit(h_A, size_A);

	for (int i = 0; i < WIDTH_A; i++) {
		for (int j = 0; j < HEIGHT_A; j++) {
			if (i < 15 || j < 15 || i > WIDTH_A - 2 || j > HEIGHT_A - 2) {
				h_A[j * WIDTH_A + i] = 0.0f;
			}
		}
	}

	// allocate device memory
	float* d_A;
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_A, mem_size_A));

	// copy host memory to device
	CUDA_SAFE_CALL(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));

	// allocate device memory for result
	unsigned int size_C = WIDTH_C * HEIGHT_C;
	unsigned int mem_size_C = sizeof(float) * size_C;

	// allocate host memory for the result
	float* h_C = (float*) malloc(mem_size_C);

	// create and start timer
	unsigned int timer = 0;

	// compute reference solution
	float* reference = (float*) malloc(mem_size_C);
	computeGold(reference, h_A, WIDTH_A, HEIGHT_A);
	CUTBoolean res;

	{
		free(h_C);
		h_C = (float*) malloc(mem_size_C);
		float* d_C;
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_C, mem_size_C));
		// setup execution parameters
		dim3 threads(16, 16);
		dim3 grid(WIDTH_C / threads.x, HEIGHT_C / threads.y);

		CUT_SAFE_CALL(cutCreateTimer(&timer));
		cudaThreadSynchronize();
		CUT_SAFE_CALL(cutStartTimer(timer));
		// execute the kernel
		imregionmax_naive<<< grid, threads >>>(d_A, d_C, WIDTH_A);
		// stop and destroy timer
		cudaThreadSynchronize();
		CUT_SAFE_CALL(cutStopTimer(timer));

		// check if kernel execution generated and error
		CUT_CHECK_ERROR("Kernel execution failed");

		// copy result from device to host
		CUDA_SAFE_CALL(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));

		printf("imregionmax_naive Processing time: %f (ms), %f Gflops \n",
				cutGetTimerValue(timer), 2000.0 * 0 / cutGetTimerValue(timer)
						/ 1024 / 1024 / 1024);
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
		dim3 grid(WIDTH_C / threads.x, WIDTH_C / (1));

		CUT_SAFE_CALL(cutCreateTimer(&timer));
		cudaThreadSynchronize();
		CUT_SAFE_CALL(cutStartTimer(timer));
		// execute the kernel
		imregionmax_coalesced<<< grid, threads >>>(d_A, d_C, WIDTH_A);
		// stop and destroy timer
		cudaThreadSynchronize();
		CUT_SAFE_CALL(cutStopTimer(timer));

		// check if kernel execution generated and error
		CUT_CHECK_ERROR("Kernel execution failed");

		// copy result from device to host
		CUDA_SAFE_CALL(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));

		printf("imregionmax_coalesced Processing time: %f (ms), %f Gflops \n",
				cutGetTimerValue(timer), 2000.0 * 0 / cutGetTimerValue(timer)
						/ 1024 / 1024 / 1024);
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
		dim3 threads(256, 1);
		dim3 grid(WIDTH_C / threads.x, WIDTH_C / (32));

		CUT_SAFE_CALL(cutCreateTimer(&timer));
		cudaThreadSynchronize();
		CUT_SAFE_CALL(cutStartTimer(timer));
		// execute the kernel
		imregionmax_opt<<< grid, threads >>>(d_A, d_C, WIDTH_A);
		// stop and destroy timer
		cudaThreadSynchronize();
		CUT_SAFE_CALL(cutStopTimer(timer));

		// check if kernel execution generated and error
		CUT_CHECK_ERROR("Kernel execution failed");

		// copy result from device to host
		CUDA_SAFE_CALL(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));

		printf("imregionmax_opt Processing time: %f (ms), %f Gflops \n",
				cutGetTimerValue(timer), 2000.0 * 0 / cutGetTimerValue(timer)
						/ 1024 / 1024 / 1024);
		CUT_SAFE_CALL(cutDeleteTimer(timer));
		CUDA_SAFE_CALL(cudaFree(d_C));
	}
	// check result
	res = cutCompareL2fe(reference, h_C, size_C, 1e-6f);
	printf("Test %s \n", (1 == res) ? "PASSED" : "FAILED");


	// clean up memory
	free(h_A);
	free(h_C);
	free(reference);
	CUDA_SAFE_CALL(cudaFree(d_A));
}

// Allocates a matrix with random float entries.
void randomInit(float* data, int size) {
	for (int i = 0; i < size; ++i)
		data[i] = rand() / (float) RAND_MAX;
}

void printDiff(float *data1, float *data2, int width, int height) {
	int i, j, k;
	int error_count = 0;
	for (j = 0; j < height; j++) {
		for (i = 0; i < width; i++) {
			k = j * width + i;
			if (data1[k] != data2[k]) {
				printf("diff(%d,%d) CPU=%4.4f, GPU=%4.4f n", i, j, data1[k],
						data2[k]);
				error_count++;
			}
		}
	}
	printf(" nTotal Errors = %d n", error_count);
}
