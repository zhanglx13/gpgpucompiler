reduction sample
1. Naive kernel (reduction.c): the naive kernel includes two stages.
    a) the first stage (the loop with k) accumulates the data with stride 'segSize'.
    b) the second stage (the loop with i) produces the final output based on the result of first step.
    In the loop body, one thread accumulates two elements into one.
    c) naive kernel specifies the value of segSize by "#pragma gCompiler gValue segSize 262144"
        
2. Execute test_reduction.sh:
	java -cp ./lib/antlr-2.7.5.jar:./lib/cetus.jar:./lib/gcompiler.jar ece.ncsu.edu.gpucompiler.cuda.KernelDriver -raw -output=test/reduction/output test/reduction/reduction.c
    a) ece.ncsu.edu.gpucompiler.cuda.KernelDriver: main function entry.
    b) -raw: the data sharing of reduction happens for read after write.
    c) -output: specify the output folder
    d) test/reduction/reduction: input naive kernel
    During the execution
    a) the compiler will first read the naive kernel and parse it.
    b) apply the coalesced pass and generate intermediate output. Nothing to do because it's already coalesced.
    c) apply the raw pass: the pass will unroll the loop and detect the read after write data sharing between different threads
        c.1) the compiler will do the thread merge first, which means it will put two threads into one thread,
            so that some memory operations (read after write) can be through the registers instead of global memory.
        c.2) then the compiler will do thread block merge, so that some memory operations (read after write) can be through the shared memory instead of global memory.
        c.3) if there are no enough resources (shared memory, registers, or thread number), the compiler will generate another kernel to finish the naive code. 
            Therefore, it will generate several kernels "gcompiler_reduction_0_output.cu", "gcompiler_reduction_1_output.cu"
    d) apply the post process and generate the final version "gcompiler_reduction_output.cu". The "gcompiler_reduction_output.cu" is only the last kernel of several kernels ("gcompiler_reduction_1_output.cu").
        So, the test case needs to use "gcompiler_reduction_0_output.cu", "gcompiler_reduction_1_output.cu" instead of gcompiler_reduction_output.cu"



3. Changes in the host code:
    blockDimX, blockDimY: the thread block configuration should be used in the host code.
    merger_y: the number that the compiler used in thread merge along the Y direction. Therefore the overall thread number in Y direction should be the original divided by merger_y.
	For example: the output of the reduction has two kernels
        a). reduction_0: 
                #define globalDimX 65536
                #define blockDimX 512
        	so that 
        	    dim3  grid(65536/512, 1, 1);
	            dim3  threads(512, 1, 1);
        b). reduction_1: 
				#define blockDimX 512
				#define globalDimX 512
        	so that 
        	    dim3  grid(1, 1, 1);
	            dim3  threads(1, 1, 1);
        
        
4. Test the code
    1. replace the reduction_opt_0 in cuda/compiler_reduction/reduction_kernel.cu with codes in "gcompiler_reduction_0_output.cu".
    1. replace the reduction_opt_1 in cuda/compiler_reduction/reduction_kernel.cu with codes in "gcompiler_reduction_1_output.cu".    
    2. update the thread block configuration in cuda/compiler_reduction/reduction.cu
    3. copy cuda/compiler_reduction to {NVIDIA SDK}/C/src/compiler_reduction
    4. go to {NVIDIA SDK}/C/src/compiler_reduction. Execute "make clean; make"
    5. Execute "../../bin/linux/release/compiler_reduction"
