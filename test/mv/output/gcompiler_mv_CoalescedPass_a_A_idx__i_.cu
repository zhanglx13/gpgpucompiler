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
#define globalDimY 1
#define A(y,x) A[(y)*WIDTH_A+(x)]
__global__ void mv(float * A, float * B, float * C, int width)
{
	__shared__ float shared_0[16][17];
	int i;
	float sum;
	sum=0;
	for (i=0; i<WIDTH_A; i=(i+16))
	{
		int it_1;
		int it_2;
		#pragma unroll 
		for (it_2=0; it_2<16; it_2=(it_2+1))
		{
			shared_0[it_2][tidx]=A(((idx+(-1*tidx))+it_2), (i+tidx));
		}
		__syncthreads();
		#pragma unroll 
		for (it_1=0; it_1<16; it_1=(it_1+1))
		{
			float a;
			float b;
			a=shared_0[tidx][it_1];
			b=B[(it_1+i)];
			sum+=(a*b);
		}
		__syncthreads();
	}
	{
		C[idx]=sum;
	}
}
