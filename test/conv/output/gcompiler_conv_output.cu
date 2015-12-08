#define COALESCED_NUM 16
#define blockDimX 256
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
	__shared__ float shared_1[16];
	__shared__ float shared_0[272];
	int i;
	int j;
	float sum = 0;
	for (j=0; j<h; j=(j+1))
	{
		for (i=0; i<w; i=(i+16))
		{
			int it_2;
			if ((tidx<16))
			{
				shared_0[(tidx+0)]=A(((idy+(( - 1)*j))+h), (((idx+(( - 1)*i))+w)+( - 16)));
			}
			shared_0[(tidx+16)]=A(((idy+(( - 1)*j))+h), ((idx+(( - 1)*i))+w));
			__syncthreads();
			if ((tidx<16))
			{
				shared_1[(tidx+0)]=B(j, ((i+0)+tidx));
			}
			__syncthreads();
			#pragma unroll 
			for (it_2=0; it_2<16; it_2=(it_2+1))
			{
				float a;
				float b;
				a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
				b=shared_1[it_2];
				sum+=(a*b);
			}
			__syncthreads();
			__syncthreads();
		}
	}
	{
		C(idy, idx)=sum;
	}
}
