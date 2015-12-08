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
#define merger_y 4
#define coalesced_idy (bidy/(COALESCED_NUM/(merger_y*blockDimY))*COALESCED_NUM)
#define A(y,x) A[(y)*WIDTH_A+(x)]
#define B(y,x) B[(y)*WIDTH_B+(x)]
#define C(y,x) C[(y)*WIDTH_C+(x)]
#define WIDTH_C 2048
#define WIDTH_B 16
#define WIDTH_A (2048+16)
__global__ void conv(float * A, float * B, float * C, int width, int height, int w, int h)
{
	__shared__ float shared_1[16][5];
	__shared__ float shared_0[272];
	int j;
	float sum_0 = 0;
	float sum_1 = 0;
	float sum_2 = 0;
	float sum_3 = 0;
	int it_2;
	for (j=0; j<(h-3); j=(j+1))
	{
		int it_2;
		if ((tidx<16))
		{
			shared_0[(tidx+0)]=A((((idy*4)+(( - 1)*j))+h), (idx+(( - 1)*0)));
		}
		shared_0[(tidx+16)]=A((((idy*4)+(( - 1)*j))+h), ((idx+(( - 1)*0))+16));
		__syncthreads();
		if ((tidx<16))
		{
			shared_1[(tidx+0)][0]=B((j+0), (0+tidx));
			shared_1[(tidx+0)][1]=B((j+1), (0+tidx));
			shared_1[(tidx+0)][2]=B((j+2), (0+tidx));
			shared_1[(tidx+0)][3]=B((j+3), (0+tidx));
		}
		__syncthreads();
		#pragma unroll 
		for (it_2=0; it_2<16; it_2=(it_2+1))
		{
			float a;
			float b_0;
			float b_1;
			float b_2;
			float b_3;
			a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
			b_0=shared_1[it_2][0];
			b_1=shared_1[it_2][1];
			b_2=shared_1[it_2][2];
			b_3=shared_1[it_2][3];
			sum_0+=(a*b_0);
			sum_1+=(a*b_1);
			sum_2+=(a*b_2);
			sum_3+=(a*b_3);
		}
		__syncthreads();
		__syncthreads();
	}
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*4)+(( - 1)*(h-1)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*4)+(( - 1)*(h-1)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][0]=B((h-1), (0+tidx));
		}
	}
	__syncthreads();
	#pragma unroll 
	for (it_2=0; it_2<16; it_2=(it_2+1))
	{
		float a;
		float b_0;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_0=shared_1[it_2][0];
		sum_0+=(a*b_0);
	}
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*4)+(( - 1)*(h-2)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*4)+(( - 1)*(h-2)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][0]=B((h-2), (0+tidx));
		}
		{
			shared_1[(tidx+0)][1]=B((h-1), (0+tidx));
		}
	}
	__syncthreads();
	#pragma unroll 
	for (it_2=0; it_2<16; it_2=(it_2+1))
	{
		float a;
		float b_0;
		float b_1;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_0=shared_1[it_2][0];
		b_1=shared_1[it_2][1];
		sum_0+=(a*b_0);
		sum_1+=(a*b_1);
	}
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*4)+(( - 1)*(h-3)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*4)+(( - 1)*(h-3)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][0]=B((h-3), (0+tidx));
		}
		{
			shared_1[(tidx+0)][1]=B((h-2), (0+tidx));
		}
		{
			shared_1[(tidx+0)][2]=B((h-1), (0+tidx));
		}
	}
	__syncthreads();
	#pragma unroll 
	for (it_2=0; it_2<16; it_2=(it_2+1))
	{
		float a;
		float b_0;
		float b_1;
		float b_2;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_0=shared_1[it_2][0];
		b_1=shared_1[it_2][1];
		b_2=shared_1[it_2][2];
		sum_0+=(a*b_0);
		sum_1+=(a*b_1);
		sum_2+=(a*b_2);
	}
	C(((idy*4)+0), idx)=sum_0;
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*4)+(( - 1)*(0-1)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*4)+(( - 1)*(0-1)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][1]=B(0, (0+tidx));
		}
		{
			shared_1[(tidx+0)][2]=B(1, (0+tidx));
		}
		{
			shared_1[(tidx+0)][3]=B(2, (0+tidx));
		}
	}
	__syncthreads();
	#pragma unroll 
	for (it_2=0; it_2<16; it_2=(it_2+1))
	{
		float a;
		float b_1;
		float b_2;
		float b_3;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_1=shared_1[it_2][1];
		b_2=shared_1[it_2][2];
		b_3=shared_1[it_2][3];
		sum_1+=(a*b_1);
		sum_2+=(a*b_2);
		sum_3+=(a*b_3);
	}
	C(((idy*4)+1), idx)=sum_1;
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*4)+(( - 1)*(0-2)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*4)+(( - 1)*(0-2)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][2]=B(0, (0+tidx));
		}
		{
			shared_1[(tidx+0)][3]=B(1, (0+tidx));
		}
	}
	__syncthreads();
	#pragma unroll 
	for (it_2=0; it_2<16; it_2=(it_2+1))
	{
		float a;
		float b_2;
		float b_3;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_2=shared_1[it_2][2];
		b_3=shared_1[it_2][3];
		sum_2+=(a*b_2);
		sum_3+=(a*b_3);
	}
	C(((idy*4)+2), idx)=sum_2;
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*4)+(( - 1)*(0-3)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*4)+(( - 1)*(0-3)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][3]=B(0, (0+tidx));
		}
	}
	__syncthreads();
	#pragma unroll 
	for (it_2=0; it_2<16; it_2=(it_2+1))
	{
		float a;
		float b_3;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_3=shared_1[it_2][3];
		sum_3+=(a*b_3);
	}
	C(((idy*4)+3), idx)=sum_3;
	__syncthreads();
	__syncthreads();
	{
		
	}
	{
		
	}
	{
		
	}
	{
		
	}
}
