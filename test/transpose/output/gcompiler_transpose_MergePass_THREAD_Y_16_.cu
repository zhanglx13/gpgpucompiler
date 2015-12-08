#define COALESCED_NUM 32
#define blockDimX 32
#define blockDimY 4
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
#define A(y,x) A[(y)*WIDTH_A+(x)]
__global__ void transpose(float * A, float * C, int width)
{
	__shared__ float shared_0[32][33];
	float sum_0 = 0;
	float sum_1 = 0;
	float sum_2 = 0;
	float sum_3 = 0;
	float sum_4 = 0;
	float sum_5 = 0;
	float sum_6 = 0;
	float sum_7 = 0;
	float sum_8 = 0;
	float sum_9 = 0;
	float sum_10 = 0;
	float sum_11 = 0;
	float sum_12 = 0;
	float sum_13 = 0;
	float sum_14 = 0;
	float sum_15 = 0;
	int it_2;
	#pragma unroll 
	for (it_2=0; it_2<32; it_2=(it_2+4))
	{
		shared_0[(it_2+(tidy*1))][tidx]=A(((idx+(( - 1)*tidx))+(it_2+(tidy*1))), (coalesced_idy+tidx));
	}
	__syncthreads();
	sum_0=shared_0[tidx][((((bidy*64)+tidy)+0)+(( - 1)*coalesced_idy))];
	sum_1=shared_0[tidx][((((bidy*64)+tidy)+4)+(( - 1)*coalesced_idy))];
	sum_2=shared_0[tidx][((((bidy*64)+tidy)+8)+(( - 1)*coalesced_idy))];
	sum_3=shared_0[tidx][((((bidy*64)+tidy)+12)+(( - 1)*coalesced_idy))];
	sum_4=shared_0[tidx][((((bidy*64)+tidy)+16)+(( - 1)*coalesced_idy))];
	sum_5=shared_0[tidx][((((bidy*64)+tidy)+20)+(( - 1)*coalesced_idy))];
	sum_6=shared_0[tidx][((((bidy*64)+tidy)+24)+(( - 1)*coalesced_idy))];
	sum_7=shared_0[tidx][((((bidy*64)+tidy)+28)+(( - 1)*coalesced_idy))];
	sum_8=shared_0[tidx][((((bidy*64)+tidy)+32)+(( - 1)*coalesced_idy))];
	sum_9=shared_0[tidx][((((bidy*64)+tidy)+36)+(( - 1)*coalesced_idy))];
	sum_10=shared_0[tidx][((((bidy*64)+tidy)+40)+(( - 1)*coalesced_idy))];
	sum_11=shared_0[tidx][((((bidy*64)+tidy)+44)+(( - 1)*coalesced_idy))];
	sum_12=shared_0[tidx][((((bidy*64)+tidy)+48)+(( - 1)*coalesced_idy))];
	sum_13=shared_0[tidx][((((bidy*64)+tidy)+52)+(( - 1)*coalesced_idy))];
	sum_14=shared_0[tidx][((((bidy*64)+tidy)+56)+(( - 1)*coalesced_idy))];
	sum_15=shared_0[tidx][((((bidy*64)+tidy)+60)+(( - 1)*coalesced_idy))];
	__syncthreads();
	__syncthreads();
	{
		C((((bidy*64)+tidy)+0), idx)=sum_0;
	}
	{
		C((((bidy*64)+tidy)+4), idx)=sum_1;
	}
	{
		C((((bidy*64)+tidy)+8), idx)=sum_2;
	}
	{
		C((((bidy*64)+tidy)+12), idx)=sum_3;
	}
	{
		C((((bidy*64)+tidy)+16), idx)=sum_4;
	}
	{
		C((((bidy*64)+tidy)+20), idx)=sum_5;
	}
	{
		C((((bidy*64)+tidy)+24), idx)=sum_6;
	}
	{
		C((((bidy*64)+tidy)+28), idx)=sum_7;
	}
	{
		C((((bidy*64)+tidy)+32), idx)=sum_8;
	}
	{
		C((((bidy*64)+tidy)+36), idx)=sum_9;
	}
	{
		C((((bidy*64)+tidy)+40), idx)=sum_10;
	}
	{
		C((((bidy*64)+tidy)+44), idx)=sum_11;
	}
	{
		C((((bidy*64)+tidy)+48), idx)=sum_12;
	}
	{
		C((((bidy*64)+tidy)+52), idx)=sum_13;
	}
	{
		C((((bidy*64)+tidy)+56), idx)=sum_14;
	}
	{
		C((((bidy*64)+tidy)+60), idx)=sum_15;
	}
}
