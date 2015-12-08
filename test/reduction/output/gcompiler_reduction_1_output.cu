#define COALESCED_NUM 16
#define blockDimX 512
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
#define globalDimX 512
#define globalDimY 1
__global__ void reduction(float * A, int size, int segSize)
{
	#pragma	gCompiler	gValue	segSize	262144
	
	__shared__ float shared_1[512];
	float tmp_4;
	float tmp_5;
	float a;
	float b;
	float c;
	{
		a=A[idx];
	}
	{
		b=A[(idx+((262144/128)/2))];
	}
	c=(a+b);
	tmp_4=c;
	{
		a=A[(idx+512)];
	}
	{
		b=A[((idx+512)+((262144/128)/2))];
	}
	c=(a+b);
	tmp_5=c;
	a=tmp_4;
	b=tmp_5;
	c=(a+b);
	shared_1[(tidx+0)]=c;
	__syncthreads();
	if ((idx<256))
	{
		float a;
		float b;
		float c;
		a=shared_1[(tidx+0)];
		b=shared_1[(tidx+256)];
		c=(a+b);
		shared_1[(tidx+0)]=c;
	}
	__syncthreads();
	if ((idx<128))
	{
		float a;
		float b;
		float c;
		a=shared_1[(tidx+0)];
		b=shared_1[(tidx+128)];
		c=(a+b);
		shared_1[(tidx+0)]=c;
	}
	__syncthreads();
	if ((idx<64))
	{
		float a;
		float b;
		float c;
		a=shared_1[(tidx+0)];
		b=shared_1[(tidx+64)];
		c=(a+b);
		shared_1[(tidx+0)]=c;
	}
	__syncthreads();
	if ((idx<32))
	{
		float a;
		float b;
		float c;
		a=shared_1[(tidx+0)];
		b=shared_1[(tidx+32)];
		c=(a+b);
		shared_1[(tidx+0)]=c;
	}
	__syncthreads();
	if ((idx<16))
	{
		float a;
		float b;
		float c;
		a=shared_1[(tidx+0)];
		b=shared_1[(tidx+16)];
		c=(a+b);
		shared_1[(tidx+0)]=c;
	}
	__syncthreads();
	if ((idx<8))
	{
		float a;
		float b;
		float c;
		a=shared_1[(tidx+0)];
		b=shared_1[(tidx+8)];
		c=(a+b);
		shared_1[(tidx+0)]=c;
	}
	__syncthreads();
	if ((idx<4))
	{
		float a;
		float b;
		float c;
		a=shared_1[(tidx+0)];
		b=shared_1[(tidx+4)];
		c=(a+b);
		shared_1[(tidx+0)]=c;
	}
	__syncthreads();
	if ((idx<2))
	{
		float a;
		float b;
		float c;
		a=shared_1[(tidx+0)];
		b=shared_1[(tidx+2)];
		c=(a+b);
		shared_1[(tidx+0)]=c;
	}
	__syncthreads();
	if ((idx<1))
	{
		float a;
		float b;
		float c;
		a=shared_1[(tidx+0)];
		b=shared_1[(tidx+1)];
		c=(a+b);
		{
			A[idx]=c;
		}
	}
}
