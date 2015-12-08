
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cublas.h>

// includes, project
#include <cutil_inline.h>

// includes, kernels
#include <vv_kernel.cu>
////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest();

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C"
void computeGold( float* a, float* b, const unsigned int len, float* result);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv)
{
    CUT_DEVICE_INIT(argc, argv);

	runTest();

    CUT_EXIT(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest()
{
    cublasStatus status;
    status = cublasInit();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! CUBLAS initialization error\n");
	exit (1);
    }

    unsigned int num_elements = WC;

    unsigned int timer;
    cutilCheckError( cutCreateTimer(&timer));

    const unsigned int in_mem_size = sizeof( float) * (num_elements);
    const unsigned int out_mem_size = sizeof( float) * (num_elements*num_elements);


    // allocate host memory to store the input data
    float* a_data = (float*) malloc( in_mem_size);
    float* b_data = (float*) malloc( in_mem_size);

    float* o_data = (float*) malloc( out_mem_size);
    float* dst_data = (float*) malloc( out_mem_size);
    float* reference = (float*) malloc( out_mem_size);

    // initialize the input data on the host to be integer values
    // between 0 and 1000
    for( unsigned int i = 0; i < num_elements; ++i)
    {
		a_data[i] = ((rand()/(float)RAND_MAX));
		b_data[i] = ((rand()/(float)RAND_MAX));
	    for( unsigned int j = 0; j < num_elements; ++j) {
	    	o_data[i*num_elements+j] = ((rand()/(float)RAND_MAX));
	    	reference[i*num_elements+j] = o_data[i*num_elements+j];
	    	dst_data[i*num_elements+j] = o_data[i*num_elements+j];
	    }


    }
//    printf("\n");

    // compute reference solution
    computeGold(a_data, b_data, num_elements, reference);

    // allocate device memory input and output arrays
    float* d_a_data;
    float* d_b_data;
    float* d_odata;
    cutilSafeCall( cudaMalloc( (void**) &d_a_data, in_mem_size));
    cutilSafeCall( cudaMalloc( (void**) &d_b_data, in_mem_size));
    cutilSafeCall( cudaMalloc( (void**) &d_odata, out_mem_size));

    // copy host memory to device input array
    cutilSafeCall( cudaMemcpy( d_a_data, a_data, in_mem_size, cudaMemcpyHostToDevice) );
    cutilSafeCall( cudaMemcpy( d_b_data, b_data, in_mem_size, cudaMemcpyHostToDevice) );

    // setup execution parameters
    // Note that these scans only support a single thread-block worth of data,
    // but we invoke them here on many blocks so that we can accurately compare
    // performance

    // make sure there are no CUDA errors before we start
    cutilCheckMsg("Kernel execution failed");

    printf("Running %d elements\n", num_elements);
	float epsilon = 1e-4;

    // execute the kernels
    unsigned int numIterations = 1;
    /*
    {

        cutilSafeCall( cudaMemcpy( d_odata, dst_data, out_mem_size, cudaMemcpyHostToDevice) );
        dim3  grid(num_elements/16, num_elements/16, 1);
        dim3  threads(16, 16, 1);
		cudaThreadSynchronize();
		cutStartTimer(timer);

		vv_naive<<< grid, threads>>> (d_a_data, d_b_data, num_elements, d_odata);
		cudaThreadSynchronize();
		cutStopTimer(timer);
		printf("Average time: %f ms; ", cutGetTimerValue(timer) / numIterations);
		cutResetTimer(timer);



		// copy result from device to host
		cutilSafeCall(cudaMemcpy( o_data, d_odata, out_mem_size,
								   cudaMemcpyDeviceToHost));


		unsigned int result_regtest = cutComparefe( reference, o_data, num_elements*num_elements, epsilon);
		printf( "warnup: Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");
    }
    */

    {

        cutilSafeCall( cudaMemcpy( d_odata, dst_data, out_mem_size, cudaMemcpyHostToDevice) );
        dim3  grid(num_elements/16, num_elements/16, 1);
        dim3  threads(16, 16, 1);
		cudaThreadSynchronize();
		cutStartTimer(timer);

		vectormul_naive<<< grid, threads>>> (d_a_data, d_b_data, d_odata, num_elements);
		cudaThreadSynchronize();
		cutStopTimer(timer);
		printf("Average time: %f ms; ", cutGetTimerValue(timer) / numIterations);
		cutResetTimer(timer);

		/*

		for (int i=0; i<16; i++) printf("%f, ", a_data[i]);
		printf("\n");
		for (int i=0; i<16; i++) printf("%f, ", b_data[i]);
		printf("\n");


		for (int i=0; i<16; i++) printf("%f, ", reference[i]);
		printf("\n");
		for (int i=0; i<16; i++) printf("%f, ", o_data[i]);
		printf("\n");
		 */

		// copy result from device to host
		cutilSafeCall(cudaMemcpy( o_data, d_odata, out_mem_size,
								   cudaMemcpyDeviceToHost));


		unsigned int result_regtest = cutComparefe( reference, o_data, num_elements*num_elements, epsilon);
		printf( "naive: Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");
    }


    {
        cutilSafeCall( cudaMemcpy( d_odata, dst_data, out_mem_size, cudaMemcpyHostToDevice) );

		cudaThreadSynchronize();
		cutStartTimer(timer);

		cublasSger(num_elements, num_elements, 1.0, d_b_data, 1, d_a_data, 1, d_odata, num_elements);
		cudaThreadSynchronize();
		cutStopTimer(timer);
		printf("Average time: %f ms; ", cutGetTimerValue(timer) / numIterations);
		cutResetTimer(timer);



		// copy result from device to host
		cutilSafeCall(cudaMemcpy( o_data, d_odata, out_mem_size,
								   cudaMemcpyDeviceToHost));



		unsigned int result_regtest = cutComparefe( reference, o_data, num_elements*num_elements, epsilon);
		printf( "cublas: Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");
    }



    {

        cutilSafeCall( cudaMemcpy( d_odata, dst_data, out_mem_size, cudaMemcpyHostToDevice) );
        dim3  grid(num_elements/32, num_elements/1, 1);
        dim3  threads(32, 1, 1);
		cudaThreadSynchronize();
		cutStartTimer(timer);

		vectormul_coalesced<<< grid, threads>>> (d_a_data, d_b_data, d_odata, num_elements);
		cudaThreadSynchronize();
		cutStopTimer(timer);
		printf("Average time: %f ms; ", cutGetTimerValue(timer) / numIterations);
		cutResetTimer(timer);

		// copy result from device to host
		cutilSafeCall(cudaMemcpy( o_data, d_odata, out_mem_size,
								   cudaMemcpyDeviceToHost));


		unsigned int result_regtest = cutComparefe( reference, o_data, num_elements*num_elements, epsilon);
		printf( "vectormul_coalesced: Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");
    }


    {
        cutilSafeCall( cudaMemcpy( d_odata, dst_data, out_mem_size, cudaMemcpyHostToDevice) );

        dim3  grid(num_elements/256, num_elements/16, 1);
        dim3  threads(256, 1, 1);
		cudaThreadSynchronize();
		cutStartTimer(timer);

		vectormul_opt<<< grid, threads>>> (d_a_data, d_b_data, d_odata, num_elements);
		cudaThreadSynchronize();
		cutStopTimer(timer);
		printf("Average time: %f ms; ", cutGetTimerValue(timer) / numIterations);
		cutResetTimer(timer);

		// copy result from device to host
		cutilSafeCall(cudaMemcpy( o_data, d_odata, out_mem_size,
								   cudaMemcpyDeviceToHost));



		unsigned int result_regtest = cutComparefe( reference, o_data, num_elements*num_elements, epsilon);
		printf( "vectormul_opt: Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");
    }




    // cleanup memory
    free( a_data);
    free( b_data);
    free( o_data);
    free( dst_data);
    free( reference);
    cutilSafeCall(cudaFree(d_a_data));
    cutilSafeCall(cudaFree(d_b_data));
    cutilSafeCall(cudaFree(d_odata));
    cutilCheckError(cutDeleteTimer(timer));
    status = cublasShutdown();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! shutdown error\n");
    }
    cudaThreadExit();

}


