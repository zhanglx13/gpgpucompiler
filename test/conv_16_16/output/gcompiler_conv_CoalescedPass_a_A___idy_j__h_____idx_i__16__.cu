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
#define A(y,x) A[(y)*WIDTH_A+(x)]
#define B(y,x) B[(y)*WIDTH_B+(x)]
#define C(y,x) C[(y)*WIDTH_C+(x)]
#define WIDTH_C 2048
#define WIDTH_B 16
#define WIDTH_A (2048+16)
__global__ void conv(float * A, float * B, float * C, int width, int height, int w, int h)
{
	__shared__ float shared_0[32];
	int j;
	float sum = 0;
	for (j=0; j<h; j=(j+1))
	{
		{
			int it_1;
			shared_0[(tidx+0)]=A(((idy+(-1*j))+h), (idx+(-1*0)));
			shared_0[(tidx+16)]=A(((idy+(-1*j))+h), ((idx+(-1*0))+16));
			__syncthreads();
			#pragma unroll 
			for (it_1=0; it_1<16; it_1=(it_1+1))
			{
				float a;
				float b;
				a=shared_0[((tidx+(-1*it_1))+16)];
				b=B(j, (it_1+0));
				sum+=(a*b);
			}
			__syncthreads();
		}
	}
	{
		C(idy, idx)=sum;
	}
}
