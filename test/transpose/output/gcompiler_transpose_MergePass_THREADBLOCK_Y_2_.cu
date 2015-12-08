#define COALESCED_NUM 32
#define blockDimX 32
#define blockDimY 2
#define gridDimX (gridDim.x)
#define gridDimY (gridDim.y)
#define idx (blockIdx.x*blockDimX+threadIdx.x)
#define idy (blockIdx.y*blockDimY+threadIdx.y)
#define bidy (blockIdx.y)
#define bidx (blockIdx.x)
#define tidx (threadIdx.x)
#define tidy (threadIdx.y)
#define merger_y 1
#define coalesced_idy (bidy/(COALESCED_NUM/(merger_y*blockDimY))*COALESCED_NUM)
#define C(y,x) C[(y)*WIDTH_C+(x)]
#define A(y,x) A[(y)*WIDTH_A+(x)]
__global__ void transpose(float * A, float * C, int width)
{
	__shared__ float shared_0[32][33];
	float sum = 0;
	int it_2;
	#pragma unroll 
	for (it_2=0; it_2<32; it_2=(it_2+2))
	{
		shared_0[(it_2+(tidy*1))][tidx]=A(((idx+(( - 1)*tidx))+(it_2+(tidy*1))), (coalesced_idy+tidx));
	}
	__syncthreads();
	sum=shared_0[tidx][(idy+(( - 1)*coalesced_idy))];
	__syncthreads();
	__syncthreads();
	{
		C(idy, idx)=sum;
	}
}
