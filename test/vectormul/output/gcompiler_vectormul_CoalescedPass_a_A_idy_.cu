#define COALESCED_NUM 16
#define blockDimX 16
#define blockDimY 1
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
__global__ void vectormul(float * A, float * B, float * C, int width)
{
	__shared__ float shared_0[16];
	float sum;
	float a;
	float b;
	sum=0;
	{
		shared_0[(tidx+0)]=A[(coalesced_idy+tidx)];
		__syncthreads();
		a=shared_0[(idy+(-1*coalesced_idy))];
		__syncthreads();
		__syncthreads();
	}
	{
		b=B[idx];
	}
	sum+=(a*b);
	{
		C(idy, idx)+=sum;
	}
}
