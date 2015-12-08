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
#define merger_y 16
#define coalesced_idy (bidy/(COALESCED_NUM/(merger_y*blockDimY))*COALESCED_NUM)
#define C(y,x) C[(y)*WIDTH_C+(x)]
__global__ void vectormul(float * A, float * B, float * C, int width)
{
	__shared__ float shared_0[16];
	float sum_0;
	float sum_1;
	float sum_2;
	float sum_3;
	float sum_4;
	float sum_5;
	float sum_6;
	float sum_7;
	float sum_8;
	float sum_9;
	float sum_10;
	float sum_11;
	float sum_12;
	float sum_13;
	float sum_14;
	float sum_15;
	float a_0;
	float a_1;
	float a_2;
	float a_3;
	float a_4;
	float a_5;
	float a_6;
	float a_7;
	float a_8;
	float a_9;
	float a_10;
	float a_11;
	float a_12;
	float a_13;
	float a_14;
	float a_15;
	float b;
	sum_0=0;
	sum_1=0;
	sum_2=0;
	sum_3=0;
	sum_4=0;
	sum_5=0;
	sum_6=0;
	sum_7=0;
	sum_8=0;
	sum_9=0;
	sum_10=0;
	sum_11=0;
	sum_12=0;
	sum_13=0;
	sum_14=0;
	sum_15=0;
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A[(coalesced_idy+tidx)];
		}
	}
	__syncthreads();
	a_0=shared_0[((((bidy*16)+tidy)+0)+(( - 1)*coalesced_idy))];
	a_1=shared_0[((((bidy*16)+tidy)+1)+(( - 1)*coalesced_idy))];
	a_2=shared_0[((((bidy*16)+tidy)+2)+(( - 1)*coalesced_idy))];
	a_3=shared_0[((((bidy*16)+tidy)+3)+(( - 1)*coalesced_idy))];
	a_4=shared_0[((((bidy*16)+tidy)+4)+(( - 1)*coalesced_idy))];
	a_5=shared_0[((((bidy*16)+tidy)+5)+(( - 1)*coalesced_idy))];
	a_6=shared_0[((((bidy*16)+tidy)+6)+(( - 1)*coalesced_idy))];
	a_7=shared_0[((((bidy*16)+tidy)+7)+(( - 1)*coalesced_idy))];
	a_8=shared_0[((((bidy*16)+tidy)+8)+(( - 1)*coalesced_idy))];
	a_9=shared_0[((((bidy*16)+tidy)+9)+(( - 1)*coalesced_idy))];
	a_10=shared_0[((((bidy*16)+tidy)+10)+(( - 1)*coalesced_idy))];
	a_11=shared_0[((((bidy*16)+tidy)+11)+(( - 1)*coalesced_idy))];
	a_12=shared_0[((((bidy*16)+tidy)+12)+(( - 1)*coalesced_idy))];
	a_13=shared_0[((((bidy*16)+tidy)+13)+(( - 1)*coalesced_idy))];
	a_14=shared_0[((((bidy*16)+tidy)+14)+(( - 1)*coalesced_idy))];
	a_15=shared_0[((((bidy*16)+tidy)+15)+(( - 1)*coalesced_idy))];
	__syncthreads();
	__syncthreads();
	{
		b=B[idx];
	}
	sum_0+=(a_0*b);
	sum_1+=(a_1*b);
	sum_2+=(a_2*b);
	sum_3+=(a_3*b);
	sum_4+=(a_4*b);
	sum_5+=(a_5*b);
	sum_6+=(a_6*b);
	sum_7+=(a_7*b);
	sum_8+=(a_8*b);
	sum_9+=(a_9*b);
	sum_10+=(a_10*b);
	sum_11+=(a_11*b);
	sum_12+=(a_12*b);
	sum_13+=(a_13*b);
	sum_14+=(a_14*b);
	sum_15+=(a_15*b);
	{
		C((((bidy*16)+tidy)+0), idx)+=sum_0;
	}
	{
		C((((bidy*16)+tidy)+1), idx)+=sum_1;
	}
	{
		C((((bidy*16)+tidy)+2), idx)+=sum_2;
	}
	{
		C((((bidy*16)+tidy)+3), idx)+=sum_3;
	}
	{
		C((((bidy*16)+tidy)+4), idx)+=sum_4;
	}
	{
		C((((bidy*16)+tidy)+5), idx)+=sum_5;
	}
	{
		C((((bidy*16)+tidy)+6), idx)+=sum_6;
	}
	{
		C((((bidy*16)+tidy)+7), idx)+=sum_7;
	}
	{
		C((((bidy*16)+tidy)+8), idx)+=sum_8;
	}
	{
		C((((bidy*16)+tidy)+9), idx)+=sum_9;
	}
	{
		C((((bidy*16)+tidy)+10), idx)+=sum_10;
	}
	{
		C((((bidy*16)+tidy)+11), idx)+=sum_11;
	}
	{
		C((((bidy*16)+tidy)+12), idx)+=sum_12;
	}
	{
		C((((bidy*16)+tidy)+13), idx)+=sum_13;
	}
	{
		C((((bidy*16)+tidy)+14), idx)+=sum_14;
	}
	{
		C((((bidy*16)+tidy)+15), idx)+=sum_15;
	}
}
