#define COALESCED_NUM 16
#define blockDimX 1
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
#define globalDimX 262144
#define globalDimY 1
__global__ void reduction_complex(float * A, float * B, int size, int segSize)
{
	#pragma	gCompiler	gValue	segSize	262144
	
	int i;
	int k;
	float sum;
	sum=0;
	for (k=0; k<size; k=(k+262144))
	{
		float real;
		float img;
		struct float2 *  tmp_0;
		tmp_0=((struct float2 * )A);
		struct float2 tmp_1;
		tmp_1=tmp_0[(idx+k)];
		real=tmp_1.x;
		img=tmp_1.y;
		sum+=real;
		sum+=img;
	}
	B[idx]=sum;
	__syncthreads();
	for (i=1; i<segSize; i=(i*2))
	{
		if ((idx<((segSize/i)/2)))
		{
			float a;
			float b;
			float c;
			a=B[idx];
			b=B[(idx+((segSize/i)/2))];
			c=(a+b);
			B[idx]=c;
		}
		__syncthreads();
	}
}
