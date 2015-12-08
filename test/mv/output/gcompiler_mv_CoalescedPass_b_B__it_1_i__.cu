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
	__shared__ float shared_1[16];
	__shared__ float shared_0[16][17];
	int i;
	float sum;
	sum=0;
	for (i=0; i<WIDTH_A; i=(i+16))
	{
		int it_2;
		#pragma unroll 
		for (it_2=0; it_2<16; it_2=(it_2+1))
		{
			shared_0[it_2][tidx]=A(((idx+(( - 1)*tidx))+it_2), (i+tidx));
		}
		__syncthreads();
		{
			int it_3;
			shared_1[(tidx+0)]=B[((i+0)+tidx)];
			__syncthreads();
			#pragma unroll 
			for (it_3=0; it_3<16; it_3=(it_3+1))
			{
				float a;
				float b;
				a=shared_0[tidx][(it_3+0)];
				b=shared_1[it_3];
				sum+=(a*b);
			}
			__syncthreads();
		}
		__syncthreads();
	}
	{
		C[idx]=sum;
	}
}
