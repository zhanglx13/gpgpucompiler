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
#define globalDimX 65536
#define globalDimY 1
__global__ void reduction(float * A, int size, int segSize)
{
	#pragma	gCompiler	gValue	segSize	262144
	
	int k;
	float sum;
	int nidx;
	__shared__ float shared_0[512];
	nidx=((((tidx/16)*2048)+(idx&15))+((idx/512)*16));
	float tmp_2;
	float tmp_3;
	float tmp_0;
	float tmp_1;
	sum=0;
	for (k=0; k<size; k=(k+262144))
	{
		float r;
		r=A[(nidx+k)];
		sum+=r;
	}
	tmp_0=sum;
	__syncthreads();
	sum=0;
	for (k=0; k<size; k=(k+262144))
	{
		float r;
		r=A[((nidx+131072)+k)];
		sum+=r;
	}
	tmp_1=sum;
	__syncthreads();
	float a;
	float b;
	float c;
	a=tmp_0;
	b=tmp_1;
	c=(a+b);
	tmp_2=c;
	sum=0;
	for (k=0; k<size; k=(k+262144))
	{
		float r;
		r=A[((nidx+65536)+k)];
		sum+=r;
	}
	tmp_0=sum;
	__syncthreads();
	sum=0;
	for (k=0; k<size; k=(k+262144))
	{
		float r;
		r=A[(((nidx+65536)+131072)+k)];
		sum+=r;
	}
	tmp_1=sum;
	__syncthreads();
	a=tmp_0;
	b=tmp_1;
	c=(a+b);
	tmp_3=c;
	a=tmp_2;
	b=tmp_3;
	c=(a+b);
	shared_0[(tidx+0)]=c;
	__syncthreads();
	if ((nidx<32768))
	{
		float a;
		float b;
		float c;
		a=shared_0[(tidx+0)];
		b=shared_0[(tidx+256)];
		c=(a+b);
		shared_0[(tidx+0)]=c;
	}
	__syncthreads();
	if ((nidx<16384))
	{
		float a;
		float b;
		float c;
		a=shared_0[(tidx+0)];
		b=shared_0[(tidx+128)];
		c=(a+b);
		shared_0[(tidx+0)]=c;
	}
	__syncthreads();
	if ((nidx<8192))
	{
		float a;
		float b;
		float c;
		a=shared_0[(tidx+0)];
		b=shared_0[(tidx+64)];
		c=(a+b);
		shared_0[(tidx+0)]=c;
	}
	__syncthreads();
	if ((nidx<4096))
	{
		float a;
		float b;
		float c;
		a=shared_0[(tidx+0)];
		b=shared_0[(tidx+32)];
		c=(a+b);
		shared_0[(tidx+0)]=c;
	}
	__syncthreads();
	if ((nidx<2048))
	{
		float a;
		float b;
		float c;
		a=shared_0[(tidx+0)];
		b=shared_0[(tidx+16)];
		c=(a+b);
		{
			A[nidx]=c;
		}
	}
}
