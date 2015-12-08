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
#define merger_y 1
#define coalesced_idy (bidy/(COALESCED_NUM/(merger_y*blockDimY))*COALESCED_NUM)
#define C(y,x) C[(y)*WIDTH_C+(x)]
#define A(y,x) A[(y)*WIDTH_A+(x)]
__global__ void imregionmax(float * A, float * C, int width)
{
	__shared__ float shared_0[144];
	float temp[9];
	int t;
	int i;
	t=0;
	#pragma unroll 
	for (i=0; i<3; i=(i+1))
	{
		int it_1;
		if ((tidx<16))
		{
			shared_0[(tidx+0)]=A(((idy+(( - 1)*i))+16), (idx+(( - 1)*0)));
		}
		shared_0[(tidx+16)]=A(((idy+(( - 1)*i))+16), ((idx+(( - 1)*0))+16));
		__syncthreads();
		#pragma unroll 
		for (it_1=0; it_1<3; it_1=(it_1+1))
		{
			float a;
			a=shared_0[((tidx+(( - 1)*it_1))+16)];
			temp[t]=a;
			t=(t+1);
		}
		__syncthreads();
	}
	{
		C(idy, idx)=cal(temp);
	}
}
