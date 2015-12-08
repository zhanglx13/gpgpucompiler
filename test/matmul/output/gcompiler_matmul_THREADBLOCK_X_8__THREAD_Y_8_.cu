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
#define merger_y 8
#define coalesced_idy (bidy/(COALESCED_NUM/(merger_y*blockDimY))*COALESCED_NUM)
#define A(y,x) A[(y)*WIDTH_A+(x)]
#define B(y,x) B[(y)*WIDTH_B+(x)]
#define C(y,x) C[(y)*WIDTH_C+(x)]
#define WIDTH_C 2048
#define WIDTH_B 2048
#define WIDTH_A 2048
__global__ void matmul(float * A, float * B, float * C, int width, int height)
{
	__shared__ float shared_0[16][9];
	int i;
	float sum_0;
	float sum_1;
	float sum_2;
	float sum_3;
	float sum_4;
	float sum_5;
	float sum_6;
	float sum_7;
	sum_0=0;
	sum_1=0;
	sum_2=0;
	sum_3=0;
	sum_4=0;
	sum_5=0;
	sum_6=0;
	sum_7=0;
	for (i=0; i<width; i=(i+16))
	{
		int it_1;
		shared_0[((tidx%16)+0)][(tidx/16)]=A((((bidy*8)+tidy)+(tidx/16)), (i+(tidx%16)));
		__syncthreads();
		#pragma unroll 
		for (it_1=0; it_1<16; it_1=(it_1+1))
		{
			float a_0;
			float a_1;
			float a_2;
			float a_3;
			float a_4;
			float a_5;
			float a_6;
			float a_7;
			float b;
			a_0=shared_0[it_1][0];
			a_1=shared_0[it_1][1];
			a_2=shared_0[it_1][2];
			a_3=shared_0[it_1][3];
			a_4=shared_0[it_1][4];
			a_5=shared_0[it_1][5];
			a_6=shared_0[it_1][6];
			a_7=shared_0[it_1][7];
			b=B((it_1+i), idx);
			sum_0+=(a_0*b);
			sum_1+=(a_1*b);
			sum_2+=(a_2*b);
			sum_3+=(a_3*b);
			sum_4+=(a_4*b);
			sum_5+=(a_5*b);
			sum_6+=(a_6*b);
			sum_7+=(a_7*b);
		}
		__syncthreads();
	}
	{
		C((((bidy*8)+tidy)+0), idx)=sum_0;
	}
	{
		C((((bidy*8)+tidy)+1), idx)=sum_1;
	}
	{
		C((((bidy*8)+tidy)+2), idx)=sum_2;
	}
	{
		C((((bidy*8)+tidy)+3), idx)=sum_3;
	}
	{
		C((((bidy*8)+tidy)+4), idx)=sum_4;
	}
	{
		C((((bidy*8)+tidy)+5), idx)=sum_5;
	}
	{
		C((((bidy*8)+tidy)+6), idx)=sum_6;
	}
	{
		C((((bidy*8)+tidy)+7), idx)=sum_7;
	}
}
