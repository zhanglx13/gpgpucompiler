matmul sample
1. Naive kernel (matmul.c)
    a) Uses macros for 2-dimensional array accesses so that the compiler can parse it as 2-dimensional array.
        For example: "#define A(y,x) A[(y)*WIDTH_A+(x)]"
    b) Define the constants such as input sizes. The source code will be used by NVCC compiler to determine the shared memory and registers usage.
        For example: "#define WIDTH_A 2048" to avoid "WIDTH_A" being treated as an unknown variable.
    c) The default thread block configuration for a naive kernel is that one thread block has 1 thread. 

        
2. Execute test_matmul.sh:
	java -cp ./lib/antlr-2.7.5.jar:./lib/cetus.jar:./lib/gcompiler.jar ece.ncsu.edu.gpucompiler.cuda.KernelDriver -cuda=cuda1_1 -output=test/matmul/output test/matmul/matmul.c
    a) ece.ncsu.edu.gpucompiler.cuda.KernelDriver: main function entry.
    b) -cuda: specify the cuda version.
    c) -output: specify the output folder
    d) test/matmul/matmul.c: input naive kernel
    During the execution
    a) the compiler will first read the naive kernel and parse it.
    b) apply the coalesced pass and generate intermediate output. For example, "gcompiler_matmul_CoalescedPass_a_A_idy__i_.cu"
    c) apply the merge pass
        c.1) the merge pass includes the data sharing in X and Y direction of thread block.
        c.2) the compiler will try to determine the number for thread (block) merge. 
        c.3) generate the intermediate versions such as "gcompiler_matmul_THREADBLOCK_X_8__THREAD_Y_16_.cu".
    d) apply the post process and generate the final version "gcompiler_matmul_output.cu".

    Other examples for compiler arguments.
    -iterator: detect data sharing between different iteration (test_conv.sh) 
    -partition: apply partition pass (test_mv.sh, test_transpose.sh)
    -raw: detect read after write data sharing between different threads (test_reduction.sh)
    -vectorization: apply vectorization pass (test_reduction_complex.sh)
    -merge1=-1:8: apply addition merge pass (test_transpose.sh)
        Users can decide the number for thread (block) merge. "-1" means let compiler choose the number. The first number is for X direction and second is for Y direction 
    

3. Changes in the host code:
    blockDimX, blockDimY: the thread block configuration should be used in the host code.
    merger_y: the number that the compiler used in thread merge along the Y direction. Therefore the overall thread number in Y direction should be the original divided by merger_y.
	For example
        1. assume the thread block configuration of your naive kernel is (1,1) for block size, (w/1,h/1) for grid size
            ***our sample codes use (16,16) for block size, (w/16,h/16) for grid size to speedup the naive kernel***
        2. after optimization, the thread block configuration is (blockDimX, blockDimY) for block size, (w/blockDimX, h/blockDimY/merger_y) for grid size
        
4. Test the code
    1. replace the matmul_opt in cuda/compiler_matmul/matrixMul_kernel.cu with codes in gcompiler_matmul_output.cu.
    2. update the thread block configuration in cuda/compiler_matmul/matrixMul.cu
    3. copy cuda/compiler_matmul to {NVIDIA SDK}/C/src/compiler_matmul
    4. go to {NVIDIA SDK}/C/src/compiler_matmul. Execute "make clean; make"
    5. Execute "../../bin/linux/release/compiler_matrixMul"
