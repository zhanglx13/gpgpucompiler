#define COALESCED_NUM 16
#define blockDimX 128
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
#define A(y,x) A[(y)*WIDTH_A+(x)]
#define B(y,x) B[(y)*WIDTH_B+(x)]
#define C(y,x) C[(y)*WIDTH_C+(x)]
#define WIDTH_C 2048
#define WIDTH_B 2048
#define WIDTH_A 2048
__global__ void matmul(float * A, float * B, float * C, int width, int height)
{
	__shared__ float shared_0[16];
	int i;
	float sum;
	sum=0;
	for (i=0; i<width; i=(i+16))
	{
		int it_1;
		if ((tidx<16))
		{
			shared_0[(tidx+0)]=A(idy, (i+tidx));
		}
		__syncthreads();
		#pragma unroll 
		for (it_1=0; it_1<16; it_1=(it_1+1))
		{
			float a;
			float b;
			a=shared_0[it_1];
			b=B((it_1+i), idx);
			sum+=(a*b);
		}
		__syncthreads();
	}
	{
		C(idy, idx)=sum;
	}
}
