Introduction:
The compiler takes naive kernels and generates optimized kernels with the following steps:
1. Preprocess: process macros and parse memory statements and loops 
2. VectorizationPass: apply vectorization 
3. CoalescedPass: code conversion for coalesced memory accesses
4. MergePass: thread (block) merge (aka loop unrolling and tiling)
4a. IteratorPass: Detecting data sharing among different iterations of different thread blocks
4b. RAWPass: Detect data sharing read after write dependence among different threads
5. PartitionPass: remove partition conflicts 
6. PrefetchingPass: apply prefetching on kernels
8. Postprocess: generate the final runnable code. 


Naive kernel:
We use the following macros to simplify the naive kernels. These macros include:
#define blockDimX 32
#define blockDimY 1
#define gridDimX (gridDim.x)
#define gridDimY (gridDim.y)
#define idx (blockIdx.x*blockDimX+threadIdx.x)
#define idy (blockIdx.y*blockDimY+threadIdx.y)
#define bidy (blockIdx.y)
#define bidx (blockIdx.x)
#define tidx (threadIdx.x)
#define tidy (threadIdx.y)
#define COALESCED_NUM 16           ---naive kernel can specify the coalesced thread number and default is 16.
#define globalDimY 1               ---naive kernel can be one dimension.
#define A(y,x) A[(y)*WIDTH_A+(x)]  ---naive kernel uses macro to map 2D array
#define merger_y 1                 ---when the compiler does thread merge in Y direction, it will set the number
#define coalesced_idy (bidy/(COALESCED_NUM/(merger_y*blockDimY))*COALESCED_NUM
                                   ---when compiler changes the thread block configure in Y direction, it will set the number

    
Supported arguments:
partition       --apply partition pass
iterator        --apply iterator pass
vectorization   --apply vectorization pass
merge0          --apply merge pass, this pass will be applied automatically.
merge1          --apply additional merge pass
prefetching     --apply prefetching pass
raw             --apply read after write pass
temp            --temp folder
output          --output folder
cuda            --CUDA version



Install
    Refer to install.txt
    
Sample
    Refer to matmul_readme.txt    