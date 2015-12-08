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
#define A(y,x) A[(y)*WIDTH_A+(x)]
#define B(y,x) B[(y)*WIDTH_B+(x)]
#define C(y,x) C[(y)*WIDTH_C+(x)]
#define WIDTH_C 2048
#define WIDTH_B 16
#define WIDTH_A (2048+16)
__global__ void conv(float * A, float * B, float * C, int width, int height, int w, int h)
{
	__shared__ float shared_1[16][17];
	__shared__ float shared_0[272];
	int j;
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
	for (j=0; j<(h-15); j=(j+1))
	{
		int it_2;
		if ((tidx<16))
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*j))+h), (idx+(( - 1)*0)));
		}
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*j))+h), ((idx+(( - 1)*0))+16));
		__syncthreads();
		if ((tidx<16))
		{
			shared_1[(tidx+0)][0]=B((j+0), (0+tidx));
			shared_1[(tidx+0)][1]=B((j+1), (0+tidx));
			shared_1[(tidx+0)][2]=B((j+2), (0+tidx));
			shared_1[(tidx+0)][3]=B((j+3), (0+tidx));
			shared_1[(tidx+0)][4]=B((j+4), (0+tidx));
			shared_1[(tidx+0)][5]=B((j+5), (0+tidx));
			shared_1[(tidx+0)][6]=B((j+6), (0+tidx));
			shared_1[(tidx+0)][7]=B((j+7), (0+tidx));
			shared_1[(tidx+0)][8]=B((j+8), (0+tidx));
			shared_1[(tidx+0)][9]=B((j+9), (0+tidx));
			shared_1[(tidx+0)][10]=B((j+10), (0+tidx));
			shared_1[(tidx+0)][11]=B((j+11), (0+tidx));
			shared_1[(tidx+0)][12]=B((j+12), (0+tidx));
			shared_1[(tidx+0)][13]=B((j+13), (0+tidx));
			shared_1[(tidx+0)][14]=B((j+14), (0+tidx));
			shared_1[(tidx+0)][15]=B((j+15), (0+tidx));
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
			float b_4;
			float b_5;
			float b_6;
			float b_7;
			float b_8;
			float b_9;
			float b_10;
			float b_11;
			float b_12;
			float b_13;
			float b_14;
			float b_15;
			a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
			b_0=shared_1[it_2][0];
			b_1=shared_1[it_2][1];
			b_2=shared_1[it_2][2];
			b_3=shared_1[it_2][3];
			b_4=shared_1[it_2][4];
			b_5=shared_1[it_2][5];
			b_6=shared_1[it_2][6];
			b_7=shared_1[it_2][7];
			b_8=shared_1[it_2][8];
			b_9=shared_1[it_2][9];
			b_10=shared_1[it_2][10];
			b_11=shared_1[it_2][11];
			b_12=shared_1[it_2][12];
			b_13=shared_1[it_2][13];
			b_14=shared_1[it_2][14];
			b_15=shared_1[it_2][15];
			sum_0+=(a*b_0);
			sum_1+=(a*b_1);
			sum_2+=(a*b_2);
			sum_3+=(a*b_3);
			sum_4+=(a*b_4);
			sum_5+=(a*b_5);
			sum_6+=(a*b_6);
			sum_7+=(a*b_7);
			sum_8+=(a*b_8);
			sum_9+=(a*b_9);
			sum_10+=(a*b_10);
			sum_11+=(a*b_11);
			sum_12+=(a*b_12);
			sum_13+=(a*b_13);
			sum_14+=(a*b_14);
			sum_15+=(a*b_15);
		}
		__syncthreads();
		__syncthreads();
	}
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(h-1)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(h-1)))+h), ((idx+(( - 1)*0))+16));
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
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(h-2)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(h-2)))+h), ((idx+(( - 1)*0))+16));
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
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(h-3)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(h-3)))+h), ((idx+(( - 1)*0))+16));
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
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(h-4)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(h-4)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][0]=B((h-4), (0+tidx));
		}
		{
			shared_1[(tidx+0)][1]=B((h-3), (0+tidx));
		}
		{
			shared_1[(tidx+0)][2]=B((h-2), (0+tidx));
		}
		{
			shared_1[(tidx+0)][3]=B((h-1), (0+tidx));
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
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(h-5)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(h-5)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][0]=B((h-5), (0+tidx));
		}
		{
			shared_1[(tidx+0)][1]=B((h-4), (0+tidx));
		}
		{
			shared_1[(tidx+0)][2]=B((h-3), (0+tidx));
		}
		{
			shared_1[(tidx+0)][3]=B((h-2), (0+tidx));
		}
		{
			shared_1[(tidx+0)][4]=B((h-1), (0+tidx));
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
		float b_3;
		float b_4;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_0=shared_1[it_2][0];
		b_1=shared_1[it_2][1];
		b_2=shared_1[it_2][2];
		b_3=shared_1[it_2][3];
		b_4=shared_1[it_2][4];
		sum_0+=(a*b_0);
		sum_1+=(a*b_1);
		sum_2+=(a*b_2);
		sum_3+=(a*b_3);
		sum_4+=(a*b_4);
	}
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(h-6)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(h-6)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][0]=B((h-6), (0+tidx));
		}
		{
			shared_1[(tidx+0)][1]=B((h-5), (0+tidx));
		}
		{
			shared_1[(tidx+0)][2]=B((h-4), (0+tidx));
		}
		{
			shared_1[(tidx+0)][3]=B((h-3), (0+tidx));
		}
		{
			shared_1[(tidx+0)][4]=B((h-2), (0+tidx));
		}
		{
			shared_1[(tidx+0)][5]=B((h-1), (0+tidx));
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
		float b_3;
		float b_4;
		float b_5;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_0=shared_1[it_2][0];
		b_1=shared_1[it_2][1];
		b_2=shared_1[it_2][2];
		b_3=shared_1[it_2][3];
		b_4=shared_1[it_2][4];
		b_5=shared_1[it_2][5];
		sum_0+=(a*b_0);
		sum_1+=(a*b_1);
		sum_2+=(a*b_2);
		sum_3+=(a*b_3);
		sum_4+=(a*b_4);
		sum_5+=(a*b_5);
	}
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(h-7)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(h-7)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][0]=B((h-7), (0+tidx));
		}
		{
			shared_1[(tidx+0)][1]=B((h-6), (0+tidx));
		}
		{
			shared_1[(tidx+0)][2]=B((h-5), (0+tidx));
		}
		{
			shared_1[(tidx+0)][3]=B((h-4), (0+tidx));
		}
		{
			shared_1[(tidx+0)][4]=B((h-3), (0+tidx));
		}
		{
			shared_1[(tidx+0)][5]=B((h-2), (0+tidx));
		}
		{
			shared_1[(tidx+0)][6]=B((h-1), (0+tidx));
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
		float b_3;
		float b_4;
		float b_5;
		float b_6;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_0=shared_1[it_2][0];
		b_1=shared_1[it_2][1];
		b_2=shared_1[it_2][2];
		b_3=shared_1[it_2][3];
		b_4=shared_1[it_2][4];
		b_5=shared_1[it_2][5];
		b_6=shared_1[it_2][6];
		sum_0+=(a*b_0);
		sum_1+=(a*b_1);
		sum_2+=(a*b_2);
		sum_3+=(a*b_3);
		sum_4+=(a*b_4);
		sum_5+=(a*b_5);
		sum_6+=(a*b_6);
	}
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(h-8)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(h-8)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][0]=B((h-8), (0+tidx));
		}
		{
			shared_1[(tidx+0)][1]=B((h-7), (0+tidx));
		}
		{
			shared_1[(tidx+0)][2]=B((h-6), (0+tidx));
		}
		{
			shared_1[(tidx+0)][3]=B((h-5), (0+tidx));
		}
		{
			shared_1[(tidx+0)][4]=B((h-4), (0+tidx));
		}
		{
			shared_1[(tidx+0)][5]=B((h-3), (0+tidx));
		}
		{
			shared_1[(tidx+0)][6]=B((h-2), (0+tidx));
		}
		{
			shared_1[(tidx+0)][7]=B((h-1), (0+tidx));
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
		float b_3;
		float b_4;
		float b_5;
		float b_6;
		float b_7;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_0=shared_1[it_2][0];
		b_1=shared_1[it_2][1];
		b_2=shared_1[it_2][2];
		b_3=shared_1[it_2][3];
		b_4=shared_1[it_2][4];
		b_5=shared_1[it_2][5];
		b_6=shared_1[it_2][6];
		b_7=shared_1[it_2][7];
		sum_0+=(a*b_0);
		sum_1+=(a*b_1);
		sum_2+=(a*b_2);
		sum_3+=(a*b_3);
		sum_4+=(a*b_4);
		sum_5+=(a*b_5);
		sum_6+=(a*b_6);
		sum_7+=(a*b_7);
	}
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(h-9)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(h-9)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][0]=B((h-9), (0+tidx));
		}
		{
			shared_1[(tidx+0)][1]=B((h-8), (0+tidx));
		}
		{
			shared_1[(tidx+0)][2]=B((h-7), (0+tidx));
		}
		{
			shared_1[(tidx+0)][3]=B((h-6), (0+tidx));
		}
		{
			shared_1[(tidx+0)][4]=B((h-5), (0+tidx));
		}
		{
			shared_1[(tidx+0)][5]=B((h-4), (0+tidx));
		}
		{
			shared_1[(tidx+0)][6]=B((h-3), (0+tidx));
		}
		{
			shared_1[(tidx+0)][7]=B((h-2), (0+tidx));
		}
		{
			shared_1[(tidx+0)][8]=B((h-1), (0+tidx));
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
		float b_3;
		float b_4;
		float b_5;
		float b_6;
		float b_7;
		float b_8;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_0=shared_1[it_2][0];
		b_1=shared_1[it_2][1];
		b_2=shared_1[it_2][2];
		b_3=shared_1[it_2][3];
		b_4=shared_1[it_2][4];
		b_5=shared_1[it_2][5];
		b_6=shared_1[it_2][6];
		b_7=shared_1[it_2][7];
		b_8=shared_1[it_2][8];
		sum_0+=(a*b_0);
		sum_1+=(a*b_1);
		sum_2+=(a*b_2);
		sum_3+=(a*b_3);
		sum_4+=(a*b_4);
		sum_5+=(a*b_5);
		sum_6+=(a*b_6);
		sum_7+=(a*b_7);
		sum_8+=(a*b_8);
	}
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(h-10)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(h-10)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][0]=B((h-10), (0+tidx));
		}
		{
			shared_1[(tidx+0)][1]=B((h-9), (0+tidx));
		}
		{
			shared_1[(tidx+0)][2]=B((h-8), (0+tidx));
		}
		{
			shared_1[(tidx+0)][3]=B((h-7), (0+tidx));
		}
		{
			shared_1[(tidx+0)][4]=B((h-6), (0+tidx));
		}
		{
			shared_1[(tidx+0)][5]=B((h-5), (0+tidx));
		}
		{
			shared_1[(tidx+0)][6]=B((h-4), (0+tidx));
		}
		{
			shared_1[(tidx+0)][7]=B((h-3), (0+tidx));
		}
		{
			shared_1[(tidx+0)][8]=B((h-2), (0+tidx));
		}
		{
			shared_1[(tidx+0)][9]=B((h-1), (0+tidx));
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
		float b_3;
		float b_4;
		float b_5;
		float b_6;
		float b_7;
		float b_8;
		float b_9;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_0=shared_1[it_2][0];
		b_1=shared_1[it_2][1];
		b_2=shared_1[it_2][2];
		b_3=shared_1[it_2][3];
		b_4=shared_1[it_2][4];
		b_5=shared_1[it_2][5];
		b_6=shared_1[it_2][6];
		b_7=shared_1[it_2][7];
		b_8=shared_1[it_2][8];
		b_9=shared_1[it_2][9];
		sum_0+=(a*b_0);
		sum_1+=(a*b_1);
		sum_2+=(a*b_2);
		sum_3+=(a*b_3);
		sum_4+=(a*b_4);
		sum_5+=(a*b_5);
		sum_6+=(a*b_6);
		sum_7+=(a*b_7);
		sum_8+=(a*b_8);
		sum_9+=(a*b_9);
	}
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(h-11)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(h-11)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][0]=B((h-11), (0+tidx));
		}
		{
			shared_1[(tidx+0)][1]=B((h-10), (0+tidx));
		}
		{
			shared_1[(tidx+0)][2]=B((h-9), (0+tidx));
		}
		{
			shared_1[(tidx+0)][3]=B((h-8), (0+tidx));
		}
		{
			shared_1[(tidx+0)][4]=B((h-7), (0+tidx));
		}
		{
			shared_1[(tidx+0)][5]=B((h-6), (0+tidx));
		}
		{
			shared_1[(tidx+0)][6]=B((h-5), (0+tidx));
		}
		{
			shared_1[(tidx+0)][7]=B((h-4), (0+tidx));
		}
		{
			shared_1[(tidx+0)][8]=B((h-3), (0+tidx));
		}
		{
			shared_1[(tidx+0)][9]=B((h-2), (0+tidx));
		}
		{
			shared_1[(tidx+0)][10]=B((h-1), (0+tidx));
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
		float b_3;
		float b_4;
		float b_5;
		float b_6;
		float b_7;
		float b_8;
		float b_9;
		float b_10;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_0=shared_1[it_2][0];
		b_1=shared_1[it_2][1];
		b_2=shared_1[it_2][2];
		b_3=shared_1[it_2][3];
		b_4=shared_1[it_2][4];
		b_5=shared_1[it_2][5];
		b_6=shared_1[it_2][6];
		b_7=shared_1[it_2][7];
		b_8=shared_1[it_2][8];
		b_9=shared_1[it_2][9];
		b_10=shared_1[it_2][10];
		sum_0+=(a*b_0);
		sum_1+=(a*b_1);
		sum_2+=(a*b_2);
		sum_3+=(a*b_3);
		sum_4+=(a*b_4);
		sum_5+=(a*b_5);
		sum_6+=(a*b_6);
		sum_7+=(a*b_7);
		sum_8+=(a*b_8);
		sum_9+=(a*b_9);
		sum_10+=(a*b_10);
	}
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(h-12)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(h-12)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][0]=B((h-12), (0+tidx));
		}
		{
			shared_1[(tidx+0)][1]=B((h-11), (0+tidx));
		}
		{
			shared_1[(tidx+0)][2]=B((h-10), (0+tidx));
		}
		{
			shared_1[(tidx+0)][3]=B((h-9), (0+tidx));
		}
		{
			shared_1[(tidx+0)][4]=B((h-8), (0+tidx));
		}
		{
			shared_1[(tidx+0)][5]=B((h-7), (0+tidx));
		}
		{
			shared_1[(tidx+0)][6]=B((h-6), (0+tidx));
		}
		{
			shared_1[(tidx+0)][7]=B((h-5), (0+tidx));
		}
		{
			shared_1[(tidx+0)][8]=B((h-4), (0+tidx));
		}
		{
			shared_1[(tidx+0)][9]=B((h-3), (0+tidx));
		}
		{
			shared_1[(tidx+0)][10]=B((h-2), (0+tidx));
		}
		{
			shared_1[(tidx+0)][11]=B((h-1), (0+tidx));
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
		float b_3;
		float b_4;
		float b_5;
		float b_6;
		float b_7;
		float b_8;
		float b_9;
		float b_10;
		float b_11;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_0=shared_1[it_2][0];
		b_1=shared_1[it_2][1];
		b_2=shared_1[it_2][2];
		b_3=shared_1[it_2][3];
		b_4=shared_1[it_2][4];
		b_5=shared_1[it_2][5];
		b_6=shared_1[it_2][6];
		b_7=shared_1[it_2][7];
		b_8=shared_1[it_2][8];
		b_9=shared_1[it_2][9];
		b_10=shared_1[it_2][10];
		b_11=shared_1[it_2][11];
		sum_0+=(a*b_0);
		sum_1+=(a*b_1);
		sum_2+=(a*b_2);
		sum_3+=(a*b_3);
		sum_4+=(a*b_4);
		sum_5+=(a*b_5);
		sum_6+=(a*b_6);
		sum_7+=(a*b_7);
		sum_8+=(a*b_8);
		sum_9+=(a*b_9);
		sum_10+=(a*b_10);
		sum_11+=(a*b_11);
	}
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(h-13)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(h-13)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][0]=B((h-13), (0+tidx));
		}
		{
			shared_1[(tidx+0)][1]=B((h-12), (0+tidx));
		}
		{
			shared_1[(tidx+0)][2]=B((h-11), (0+tidx));
		}
		{
			shared_1[(tidx+0)][3]=B((h-10), (0+tidx));
		}
		{
			shared_1[(tidx+0)][4]=B((h-9), (0+tidx));
		}
		{
			shared_1[(tidx+0)][5]=B((h-8), (0+tidx));
		}
		{
			shared_1[(tidx+0)][6]=B((h-7), (0+tidx));
		}
		{
			shared_1[(tidx+0)][7]=B((h-6), (0+tidx));
		}
		{
			shared_1[(tidx+0)][8]=B((h-5), (0+tidx));
		}
		{
			shared_1[(tidx+0)][9]=B((h-4), (0+tidx));
		}
		{
			shared_1[(tidx+0)][10]=B((h-3), (0+tidx));
		}
		{
			shared_1[(tidx+0)][11]=B((h-2), (0+tidx));
		}
		{
			shared_1[(tidx+0)][12]=B((h-1), (0+tidx));
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
		float b_3;
		float b_4;
		float b_5;
		float b_6;
		float b_7;
		float b_8;
		float b_9;
		float b_10;
		float b_11;
		float b_12;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_0=shared_1[it_2][0];
		b_1=shared_1[it_2][1];
		b_2=shared_1[it_2][2];
		b_3=shared_1[it_2][3];
		b_4=shared_1[it_2][4];
		b_5=shared_1[it_2][5];
		b_6=shared_1[it_2][6];
		b_7=shared_1[it_2][7];
		b_8=shared_1[it_2][8];
		b_9=shared_1[it_2][9];
		b_10=shared_1[it_2][10];
		b_11=shared_1[it_2][11];
		b_12=shared_1[it_2][12];
		sum_0+=(a*b_0);
		sum_1+=(a*b_1);
		sum_2+=(a*b_2);
		sum_3+=(a*b_3);
		sum_4+=(a*b_4);
		sum_5+=(a*b_5);
		sum_6+=(a*b_6);
		sum_7+=(a*b_7);
		sum_8+=(a*b_8);
		sum_9+=(a*b_9);
		sum_10+=(a*b_10);
		sum_11+=(a*b_11);
		sum_12+=(a*b_12);
	}
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(h-14)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(h-14)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][0]=B((h-14), (0+tidx));
		}
		{
			shared_1[(tidx+0)][1]=B((h-13), (0+tidx));
		}
		{
			shared_1[(tidx+0)][2]=B((h-12), (0+tidx));
		}
		{
			shared_1[(tidx+0)][3]=B((h-11), (0+tidx));
		}
		{
			shared_1[(tidx+0)][4]=B((h-10), (0+tidx));
		}
		{
			shared_1[(tidx+0)][5]=B((h-9), (0+tidx));
		}
		{
			shared_1[(tidx+0)][6]=B((h-8), (0+tidx));
		}
		{
			shared_1[(tidx+0)][7]=B((h-7), (0+tidx));
		}
		{
			shared_1[(tidx+0)][8]=B((h-6), (0+tidx));
		}
		{
			shared_1[(tidx+0)][9]=B((h-5), (0+tidx));
		}
		{
			shared_1[(tidx+0)][10]=B((h-4), (0+tidx));
		}
		{
			shared_1[(tidx+0)][11]=B((h-3), (0+tidx));
		}
		{
			shared_1[(tidx+0)][12]=B((h-2), (0+tidx));
		}
		{
			shared_1[(tidx+0)][13]=B((h-1), (0+tidx));
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
		float b_3;
		float b_4;
		float b_5;
		float b_6;
		float b_7;
		float b_8;
		float b_9;
		float b_10;
		float b_11;
		float b_12;
		float b_13;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_0=shared_1[it_2][0];
		b_1=shared_1[it_2][1];
		b_2=shared_1[it_2][2];
		b_3=shared_1[it_2][3];
		b_4=shared_1[it_2][4];
		b_5=shared_1[it_2][5];
		b_6=shared_1[it_2][6];
		b_7=shared_1[it_2][7];
		b_8=shared_1[it_2][8];
		b_9=shared_1[it_2][9];
		b_10=shared_1[it_2][10];
		b_11=shared_1[it_2][11];
		b_12=shared_1[it_2][12];
		b_13=shared_1[it_2][13];
		sum_0+=(a*b_0);
		sum_1+=(a*b_1);
		sum_2+=(a*b_2);
		sum_3+=(a*b_3);
		sum_4+=(a*b_4);
		sum_5+=(a*b_5);
		sum_6+=(a*b_6);
		sum_7+=(a*b_7);
		sum_8+=(a*b_8);
		sum_9+=(a*b_9);
		sum_10+=(a*b_10);
		sum_11+=(a*b_11);
		sum_12+=(a*b_12);
		sum_13+=(a*b_13);
	}
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(h-15)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(h-15)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][0]=B((h-15), (0+tidx));
		}
		{
			shared_1[(tidx+0)][1]=B((h-14), (0+tidx));
		}
		{
			shared_1[(tidx+0)][2]=B((h-13), (0+tidx));
		}
		{
			shared_1[(tidx+0)][3]=B((h-12), (0+tidx));
		}
		{
			shared_1[(tidx+0)][4]=B((h-11), (0+tidx));
		}
		{
			shared_1[(tidx+0)][5]=B((h-10), (0+tidx));
		}
		{
			shared_1[(tidx+0)][6]=B((h-9), (0+tidx));
		}
		{
			shared_1[(tidx+0)][7]=B((h-8), (0+tidx));
		}
		{
			shared_1[(tidx+0)][8]=B((h-7), (0+tidx));
		}
		{
			shared_1[(tidx+0)][9]=B((h-6), (0+tidx));
		}
		{
			shared_1[(tidx+0)][10]=B((h-5), (0+tidx));
		}
		{
			shared_1[(tidx+0)][11]=B((h-4), (0+tidx));
		}
		{
			shared_1[(tidx+0)][12]=B((h-3), (0+tidx));
		}
		{
			shared_1[(tidx+0)][13]=B((h-2), (0+tidx));
		}
		{
			shared_1[(tidx+0)][14]=B((h-1), (0+tidx));
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
		float b_3;
		float b_4;
		float b_5;
		float b_6;
		float b_7;
		float b_8;
		float b_9;
		float b_10;
		float b_11;
		float b_12;
		float b_13;
		float b_14;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_0=shared_1[it_2][0];
		b_1=shared_1[it_2][1];
		b_2=shared_1[it_2][2];
		b_3=shared_1[it_2][3];
		b_4=shared_1[it_2][4];
		b_5=shared_1[it_2][5];
		b_6=shared_1[it_2][6];
		b_7=shared_1[it_2][7];
		b_8=shared_1[it_2][8];
		b_9=shared_1[it_2][9];
		b_10=shared_1[it_2][10];
		b_11=shared_1[it_2][11];
		b_12=shared_1[it_2][12];
		b_13=shared_1[it_2][13];
		b_14=shared_1[it_2][14];
		sum_0+=(a*b_0);
		sum_1+=(a*b_1);
		sum_2+=(a*b_2);
		sum_3+=(a*b_3);
		sum_4+=(a*b_4);
		sum_5+=(a*b_5);
		sum_6+=(a*b_6);
		sum_7+=(a*b_7);
		sum_8+=(a*b_8);
		sum_9+=(a*b_9);
		sum_10+=(a*b_10);
		sum_11+=(a*b_11);
		sum_12+=(a*b_12);
		sum_13+=(a*b_13);
		sum_14+=(a*b_14);
	}
	C(((idy*16)+0), idx)=sum_0;
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(0-1)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(0-1)))+h), ((idx+(( - 1)*0))+16));
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
		{
			shared_1[(tidx+0)][4]=B(3, (0+tidx));
		}
		{
			shared_1[(tidx+0)][5]=B(4, (0+tidx));
		}
		{
			shared_1[(tidx+0)][6]=B(5, (0+tidx));
		}
		{
			shared_1[(tidx+0)][7]=B(6, (0+tidx));
		}
		{
			shared_1[(tidx+0)][8]=B(7, (0+tidx));
		}
		{
			shared_1[(tidx+0)][9]=B(8, (0+tidx));
		}
		{
			shared_1[(tidx+0)][10]=B(9, (0+tidx));
		}
		{
			shared_1[(tidx+0)][11]=B(10, (0+tidx));
		}
		{
			shared_1[(tidx+0)][12]=B(11, (0+tidx));
		}
		{
			shared_1[(tidx+0)][13]=B(12, (0+tidx));
		}
		{
			shared_1[(tidx+0)][14]=B(13, (0+tidx));
		}
		{
			shared_1[(tidx+0)][15]=B(14, (0+tidx));
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
		float b_4;
		float b_5;
		float b_6;
		float b_7;
		float b_8;
		float b_9;
		float b_10;
		float b_11;
		float b_12;
		float b_13;
		float b_14;
		float b_15;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_1=shared_1[it_2][1];
		b_2=shared_1[it_2][2];
		b_3=shared_1[it_2][3];
		b_4=shared_1[it_2][4];
		b_5=shared_1[it_2][5];
		b_6=shared_1[it_2][6];
		b_7=shared_1[it_2][7];
		b_8=shared_1[it_2][8];
		b_9=shared_1[it_2][9];
		b_10=shared_1[it_2][10];
		b_11=shared_1[it_2][11];
		b_12=shared_1[it_2][12];
		b_13=shared_1[it_2][13];
		b_14=shared_1[it_2][14];
		b_15=shared_1[it_2][15];
		sum_1+=(a*b_1);
		sum_2+=(a*b_2);
		sum_3+=(a*b_3);
		sum_4+=(a*b_4);
		sum_5+=(a*b_5);
		sum_6+=(a*b_6);
		sum_7+=(a*b_7);
		sum_8+=(a*b_8);
		sum_9+=(a*b_9);
		sum_10+=(a*b_10);
		sum_11+=(a*b_11);
		sum_12+=(a*b_12);
		sum_13+=(a*b_13);
		sum_14+=(a*b_14);
		sum_15+=(a*b_15);
	}
	C(((idy*16)+1), idx)=sum_1;
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(0-2)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(0-2)))+h), ((idx+(( - 1)*0))+16));
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
		{
			shared_1[(tidx+0)][4]=B(2, (0+tidx));
		}
		{
			shared_1[(tidx+0)][5]=B(3, (0+tidx));
		}
		{
			shared_1[(tidx+0)][6]=B(4, (0+tidx));
		}
		{
			shared_1[(tidx+0)][7]=B(5, (0+tidx));
		}
		{
			shared_1[(tidx+0)][8]=B(6, (0+tidx));
		}
		{
			shared_1[(tidx+0)][9]=B(7, (0+tidx));
		}
		{
			shared_1[(tidx+0)][10]=B(8, (0+tidx));
		}
		{
			shared_1[(tidx+0)][11]=B(9, (0+tidx));
		}
		{
			shared_1[(tidx+0)][12]=B(10, (0+tidx));
		}
		{
			shared_1[(tidx+0)][13]=B(11, (0+tidx));
		}
		{
			shared_1[(tidx+0)][14]=B(12, (0+tidx));
		}
		{
			shared_1[(tidx+0)][15]=B(13, (0+tidx));
		}
	}
	__syncthreads();
	#pragma unroll 
	for (it_2=0; it_2<16; it_2=(it_2+1))
	{
		float a;
		float b_2;
		float b_3;
		float b_4;
		float b_5;
		float b_6;
		float b_7;
		float b_8;
		float b_9;
		float b_10;
		float b_11;
		float b_12;
		float b_13;
		float b_14;
		float b_15;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_2=shared_1[it_2][2];
		b_3=shared_1[it_2][3];
		b_4=shared_1[it_2][4];
		b_5=shared_1[it_2][5];
		b_6=shared_1[it_2][6];
		b_7=shared_1[it_2][7];
		b_8=shared_1[it_2][8];
		b_9=shared_1[it_2][9];
		b_10=shared_1[it_2][10];
		b_11=shared_1[it_2][11];
		b_12=shared_1[it_2][12];
		b_13=shared_1[it_2][13];
		b_14=shared_1[it_2][14];
		b_15=shared_1[it_2][15];
		sum_2+=(a*b_2);
		sum_3+=(a*b_3);
		sum_4+=(a*b_4);
		sum_5+=(a*b_5);
		sum_6+=(a*b_6);
		sum_7+=(a*b_7);
		sum_8+=(a*b_8);
		sum_9+=(a*b_9);
		sum_10+=(a*b_10);
		sum_11+=(a*b_11);
		sum_12+=(a*b_12);
		sum_13+=(a*b_13);
		sum_14+=(a*b_14);
		sum_15+=(a*b_15);
	}
	C(((idy*16)+2), idx)=sum_2;
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(0-3)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(0-3)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][3]=B(0, (0+tidx));
		}
		{
			shared_1[(tidx+0)][4]=B(1, (0+tidx));
		}
		{
			shared_1[(tidx+0)][5]=B(2, (0+tidx));
		}
		{
			shared_1[(tidx+0)][6]=B(3, (0+tidx));
		}
		{
			shared_1[(tidx+0)][7]=B(4, (0+tidx));
		}
		{
			shared_1[(tidx+0)][8]=B(5, (0+tidx));
		}
		{
			shared_1[(tidx+0)][9]=B(6, (0+tidx));
		}
		{
			shared_1[(tidx+0)][10]=B(7, (0+tidx));
		}
		{
			shared_1[(tidx+0)][11]=B(8, (0+tidx));
		}
		{
			shared_1[(tidx+0)][12]=B(9, (0+tidx));
		}
		{
			shared_1[(tidx+0)][13]=B(10, (0+tidx));
		}
		{
			shared_1[(tidx+0)][14]=B(11, (0+tidx));
		}
		{
			shared_1[(tidx+0)][15]=B(12, (0+tidx));
		}
	}
	__syncthreads();
	#pragma unroll 
	for (it_2=0; it_2<16; it_2=(it_2+1))
	{
		float a;
		float b_3;
		float b_4;
		float b_5;
		float b_6;
		float b_7;
		float b_8;
		float b_9;
		float b_10;
		float b_11;
		float b_12;
		float b_13;
		float b_14;
		float b_15;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_3=shared_1[it_2][3];
		b_4=shared_1[it_2][4];
		b_5=shared_1[it_2][5];
		b_6=shared_1[it_2][6];
		b_7=shared_1[it_2][7];
		b_8=shared_1[it_2][8];
		b_9=shared_1[it_2][9];
		b_10=shared_1[it_2][10];
		b_11=shared_1[it_2][11];
		b_12=shared_1[it_2][12];
		b_13=shared_1[it_2][13];
		b_14=shared_1[it_2][14];
		b_15=shared_1[it_2][15];
		sum_3+=(a*b_3);
		sum_4+=(a*b_4);
		sum_5+=(a*b_5);
		sum_6+=(a*b_6);
		sum_7+=(a*b_7);
		sum_8+=(a*b_8);
		sum_9+=(a*b_9);
		sum_10+=(a*b_10);
		sum_11+=(a*b_11);
		sum_12+=(a*b_12);
		sum_13+=(a*b_13);
		sum_14+=(a*b_14);
		sum_15+=(a*b_15);
	}
	C(((idy*16)+3), idx)=sum_3;
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(0-4)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(0-4)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][4]=B(0, (0+tidx));
		}
		{
			shared_1[(tidx+0)][5]=B(1, (0+tidx));
		}
		{
			shared_1[(tidx+0)][6]=B(2, (0+tidx));
		}
		{
			shared_1[(tidx+0)][7]=B(3, (0+tidx));
		}
		{
			shared_1[(tidx+0)][8]=B(4, (0+tidx));
		}
		{
			shared_1[(tidx+0)][9]=B(5, (0+tidx));
		}
		{
			shared_1[(tidx+0)][10]=B(6, (0+tidx));
		}
		{
			shared_1[(tidx+0)][11]=B(7, (0+tidx));
		}
		{
			shared_1[(tidx+0)][12]=B(8, (0+tidx));
		}
		{
			shared_1[(tidx+0)][13]=B(9, (0+tidx));
		}
		{
			shared_1[(tidx+0)][14]=B(10, (0+tidx));
		}
		{
			shared_1[(tidx+0)][15]=B(11, (0+tidx));
		}
	}
	__syncthreads();
	#pragma unroll 
	for (it_2=0; it_2<16; it_2=(it_2+1))
	{
		float a;
		float b_4;
		float b_5;
		float b_6;
		float b_7;
		float b_8;
		float b_9;
		float b_10;
		float b_11;
		float b_12;
		float b_13;
		float b_14;
		float b_15;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_4=shared_1[it_2][4];
		b_5=shared_1[it_2][5];
		b_6=shared_1[it_2][6];
		b_7=shared_1[it_2][7];
		b_8=shared_1[it_2][8];
		b_9=shared_1[it_2][9];
		b_10=shared_1[it_2][10];
		b_11=shared_1[it_2][11];
		b_12=shared_1[it_2][12];
		b_13=shared_1[it_2][13];
		b_14=shared_1[it_2][14];
		b_15=shared_1[it_2][15];
		sum_4+=(a*b_4);
		sum_5+=(a*b_5);
		sum_6+=(a*b_6);
		sum_7+=(a*b_7);
		sum_8+=(a*b_8);
		sum_9+=(a*b_9);
		sum_10+=(a*b_10);
		sum_11+=(a*b_11);
		sum_12+=(a*b_12);
		sum_13+=(a*b_13);
		sum_14+=(a*b_14);
		sum_15+=(a*b_15);
	}
	C(((idy*16)+4), idx)=sum_4;
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(0-5)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(0-5)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][5]=B(0, (0+tidx));
		}
		{
			shared_1[(tidx+0)][6]=B(1, (0+tidx));
		}
		{
			shared_1[(tidx+0)][7]=B(2, (0+tidx));
		}
		{
			shared_1[(tidx+0)][8]=B(3, (0+tidx));
		}
		{
			shared_1[(tidx+0)][9]=B(4, (0+tidx));
		}
		{
			shared_1[(tidx+0)][10]=B(5, (0+tidx));
		}
		{
			shared_1[(tidx+0)][11]=B(6, (0+tidx));
		}
		{
			shared_1[(tidx+0)][12]=B(7, (0+tidx));
		}
		{
			shared_1[(tidx+0)][13]=B(8, (0+tidx));
		}
		{
			shared_1[(tidx+0)][14]=B(9, (0+tidx));
		}
		{
			shared_1[(tidx+0)][15]=B(10, (0+tidx));
		}
	}
	__syncthreads();
	#pragma unroll 
	for (it_2=0; it_2<16; it_2=(it_2+1))
	{
		float a;
		float b_5;
		float b_6;
		float b_7;
		float b_8;
		float b_9;
		float b_10;
		float b_11;
		float b_12;
		float b_13;
		float b_14;
		float b_15;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_5=shared_1[it_2][5];
		b_6=shared_1[it_2][6];
		b_7=shared_1[it_2][7];
		b_8=shared_1[it_2][8];
		b_9=shared_1[it_2][9];
		b_10=shared_1[it_2][10];
		b_11=shared_1[it_2][11];
		b_12=shared_1[it_2][12];
		b_13=shared_1[it_2][13];
		b_14=shared_1[it_2][14];
		b_15=shared_1[it_2][15];
		sum_5+=(a*b_5);
		sum_6+=(a*b_6);
		sum_7+=(a*b_7);
		sum_8+=(a*b_8);
		sum_9+=(a*b_9);
		sum_10+=(a*b_10);
		sum_11+=(a*b_11);
		sum_12+=(a*b_12);
		sum_13+=(a*b_13);
		sum_14+=(a*b_14);
		sum_15+=(a*b_15);
	}
	C(((idy*16)+5), idx)=sum_5;
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(0-6)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(0-6)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][6]=B(0, (0+tidx));
		}
		{
			shared_1[(tidx+0)][7]=B(1, (0+tidx));
		}
		{
			shared_1[(tidx+0)][8]=B(2, (0+tidx));
		}
		{
			shared_1[(tidx+0)][9]=B(3, (0+tidx));
		}
		{
			shared_1[(tidx+0)][10]=B(4, (0+tidx));
		}
		{
			shared_1[(tidx+0)][11]=B(5, (0+tidx));
		}
		{
			shared_1[(tidx+0)][12]=B(6, (0+tidx));
		}
		{
			shared_1[(tidx+0)][13]=B(7, (0+tidx));
		}
		{
			shared_1[(tidx+0)][14]=B(8, (0+tidx));
		}
		{
			shared_1[(tidx+0)][15]=B(9, (0+tidx));
		}
	}
	__syncthreads();
	#pragma unroll 
	for (it_2=0; it_2<16; it_2=(it_2+1))
	{
		float a;
		float b_6;
		float b_7;
		float b_8;
		float b_9;
		float b_10;
		float b_11;
		float b_12;
		float b_13;
		float b_14;
		float b_15;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_6=shared_1[it_2][6];
		b_7=shared_1[it_2][7];
		b_8=shared_1[it_2][8];
		b_9=shared_1[it_2][9];
		b_10=shared_1[it_2][10];
		b_11=shared_1[it_2][11];
		b_12=shared_1[it_2][12];
		b_13=shared_1[it_2][13];
		b_14=shared_1[it_2][14];
		b_15=shared_1[it_2][15];
		sum_6+=(a*b_6);
		sum_7+=(a*b_7);
		sum_8+=(a*b_8);
		sum_9+=(a*b_9);
		sum_10+=(a*b_10);
		sum_11+=(a*b_11);
		sum_12+=(a*b_12);
		sum_13+=(a*b_13);
		sum_14+=(a*b_14);
		sum_15+=(a*b_15);
	}
	C(((idy*16)+6), idx)=sum_6;
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(0-7)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(0-7)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][7]=B(0, (0+tidx));
		}
		{
			shared_1[(tidx+0)][8]=B(1, (0+tidx));
		}
		{
			shared_1[(tidx+0)][9]=B(2, (0+tidx));
		}
		{
			shared_1[(tidx+0)][10]=B(3, (0+tidx));
		}
		{
			shared_1[(tidx+0)][11]=B(4, (0+tidx));
		}
		{
			shared_1[(tidx+0)][12]=B(5, (0+tidx));
		}
		{
			shared_1[(tidx+0)][13]=B(6, (0+tidx));
		}
		{
			shared_1[(tidx+0)][14]=B(7, (0+tidx));
		}
		{
			shared_1[(tidx+0)][15]=B(8, (0+tidx));
		}
	}
	__syncthreads();
	#pragma unroll 
	for (it_2=0; it_2<16; it_2=(it_2+1))
	{
		float a;
		float b_7;
		float b_8;
		float b_9;
		float b_10;
		float b_11;
		float b_12;
		float b_13;
		float b_14;
		float b_15;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_7=shared_1[it_2][7];
		b_8=shared_1[it_2][8];
		b_9=shared_1[it_2][9];
		b_10=shared_1[it_2][10];
		b_11=shared_1[it_2][11];
		b_12=shared_1[it_2][12];
		b_13=shared_1[it_2][13];
		b_14=shared_1[it_2][14];
		b_15=shared_1[it_2][15];
		sum_7+=(a*b_7);
		sum_8+=(a*b_8);
		sum_9+=(a*b_9);
		sum_10+=(a*b_10);
		sum_11+=(a*b_11);
		sum_12+=(a*b_12);
		sum_13+=(a*b_13);
		sum_14+=(a*b_14);
		sum_15+=(a*b_15);
	}
	C(((idy*16)+7), idx)=sum_7;
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(0-8)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(0-8)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][8]=B(0, (0+tidx));
		}
		{
			shared_1[(tidx+0)][9]=B(1, (0+tidx));
		}
		{
			shared_1[(tidx+0)][10]=B(2, (0+tidx));
		}
		{
			shared_1[(tidx+0)][11]=B(3, (0+tidx));
		}
		{
			shared_1[(tidx+0)][12]=B(4, (0+tidx));
		}
		{
			shared_1[(tidx+0)][13]=B(5, (0+tidx));
		}
		{
			shared_1[(tidx+0)][14]=B(6, (0+tidx));
		}
		{
			shared_1[(tidx+0)][15]=B(7, (0+tidx));
		}
	}
	__syncthreads();
	#pragma unroll 
	for (it_2=0; it_2<16; it_2=(it_2+1))
	{
		float a;
		float b_8;
		float b_9;
		float b_10;
		float b_11;
		float b_12;
		float b_13;
		float b_14;
		float b_15;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_8=shared_1[it_2][8];
		b_9=shared_1[it_2][9];
		b_10=shared_1[it_2][10];
		b_11=shared_1[it_2][11];
		b_12=shared_1[it_2][12];
		b_13=shared_1[it_2][13];
		b_14=shared_1[it_2][14];
		b_15=shared_1[it_2][15];
		sum_8+=(a*b_8);
		sum_9+=(a*b_9);
		sum_10+=(a*b_10);
		sum_11+=(a*b_11);
		sum_12+=(a*b_12);
		sum_13+=(a*b_13);
		sum_14+=(a*b_14);
		sum_15+=(a*b_15);
	}
	C(((idy*16)+8), idx)=sum_8;
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(0-9)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(0-9)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][9]=B(0, (0+tidx));
		}
		{
			shared_1[(tidx+0)][10]=B(1, (0+tidx));
		}
		{
			shared_1[(tidx+0)][11]=B(2, (0+tidx));
		}
		{
			shared_1[(tidx+0)][12]=B(3, (0+tidx));
		}
		{
			shared_1[(tidx+0)][13]=B(4, (0+tidx));
		}
		{
			shared_1[(tidx+0)][14]=B(5, (0+tidx));
		}
		{
			shared_1[(tidx+0)][15]=B(6, (0+tidx));
		}
	}
	__syncthreads();
	#pragma unroll 
	for (it_2=0; it_2<16; it_2=(it_2+1))
	{
		float a;
		float b_9;
		float b_10;
		float b_11;
		float b_12;
		float b_13;
		float b_14;
		float b_15;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_9=shared_1[it_2][9];
		b_10=shared_1[it_2][10];
		b_11=shared_1[it_2][11];
		b_12=shared_1[it_2][12];
		b_13=shared_1[it_2][13];
		b_14=shared_1[it_2][14];
		b_15=shared_1[it_2][15];
		sum_9+=(a*b_9);
		sum_10+=(a*b_10);
		sum_11+=(a*b_11);
		sum_12+=(a*b_12);
		sum_13+=(a*b_13);
		sum_14+=(a*b_14);
		sum_15+=(a*b_15);
	}
	C(((idy*16)+9), idx)=sum_9;
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(0-10)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(0-10)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][10]=B(0, (0+tidx));
		}
		{
			shared_1[(tidx+0)][11]=B(1, (0+tidx));
		}
		{
			shared_1[(tidx+0)][12]=B(2, (0+tidx));
		}
		{
			shared_1[(tidx+0)][13]=B(3, (0+tidx));
		}
		{
			shared_1[(tidx+0)][14]=B(4, (0+tidx));
		}
		{
			shared_1[(tidx+0)][15]=B(5, (0+tidx));
		}
	}
	__syncthreads();
	#pragma unroll 
	for (it_2=0; it_2<16; it_2=(it_2+1))
	{
		float a;
		float b_10;
		float b_11;
		float b_12;
		float b_13;
		float b_14;
		float b_15;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_10=shared_1[it_2][10];
		b_11=shared_1[it_2][11];
		b_12=shared_1[it_2][12];
		b_13=shared_1[it_2][13];
		b_14=shared_1[it_2][14];
		b_15=shared_1[it_2][15];
		sum_10+=(a*b_10);
		sum_11+=(a*b_11);
		sum_12+=(a*b_12);
		sum_13+=(a*b_13);
		sum_14+=(a*b_14);
		sum_15+=(a*b_15);
	}
	C(((idy*16)+10), idx)=sum_10;
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(0-11)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(0-11)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][11]=B(0, (0+tidx));
		}
		{
			shared_1[(tidx+0)][12]=B(1, (0+tidx));
		}
		{
			shared_1[(tidx+0)][13]=B(2, (0+tidx));
		}
		{
			shared_1[(tidx+0)][14]=B(3, (0+tidx));
		}
		{
			shared_1[(tidx+0)][15]=B(4, (0+tidx));
		}
	}
	__syncthreads();
	#pragma unroll 
	for (it_2=0; it_2<16; it_2=(it_2+1))
	{
		float a;
		float b_11;
		float b_12;
		float b_13;
		float b_14;
		float b_15;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_11=shared_1[it_2][11];
		b_12=shared_1[it_2][12];
		b_13=shared_1[it_2][13];
		b_14=shared_1[it_2][14];
		b_15=shared_1[it_2][15];
		sum_11+=(a*b_11);
		sum_12+=(a*b_12);
		sum_13+=(a*b_13);
		sum_14+=(a*b_14);
		sum_15+=(a*b_15);
	}
	C(((idy*16)+11), idx)=sum_11;
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(0-12)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(0-12)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][12]=B(0, (0+tidx));
		}
		{
			shared_1[(tidx+0)][13]=B(1, (0+tidx));
		}
		{
			shared_1[(tidx+0)][14]=B(2, (0+tidx));
		}
		{
			shared_1[(tidx+0)][15]=B(3, (0+tidx));
		}
	}
	__syncthreads();
	#pragma unroll 
	for (it_2=0; it_2<16; it_2=(it_2+1))
	{
		float a;
		float b_12;
		float b_13;
		float b_14;
		float b_15;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_12=shared_1[it_2][12];
		b_13=shared_1[it_2][13];
		b_14=shared_1[it_2][14];
		b_15=shared_1[it_2][15];
		sum_12+=(a*b_12);
		sum_13+=(a*b_13);
		sum_14+=(a*b_14);
		sum_15+=(a*b_15);
	}
	C(((idy*16)+12), idx)=sum_12;
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(0-13)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(0-13)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][13]=B(0, (0+tidx));
		}
		{
			shared_1[(tidx+0)][14]=B(1, (0+tidx));
		}
		{
			shared_1[(tidx+0)][15]=B(2, (0+tidx));
		}
	}
	__syncthreads();
	#pragma unroll 
	for (it_2=0; it_2<16; it_2=(it_2+1))
	{
		float a;
		float b_13;
		float b_14;
		float b_15;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_13=shared_1[it_2][13];
		b_14=shared_1[it_2][14];
		b_15=shared_1[it_2][15];
		sum_13+=(a*b_13);
		sum_14+=(a*b_14);
		sum_15+=(a*b_15);
	}
	C(((idy*16)+13), idx)=sum_13;
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(0-14)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(0-14)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][14]=B(0, (0+tidx));
		}
		{
			shared_1[(tidx+0)][15]=B(1, (0+tidx));
		}
	}
	__syncthreads();
	#pragma unroll 
	for (it_2=0; it_2<16; it_2=(it_2+1))
	{
		float a;
		float b_14;
		float b_15;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_14=shared_1[it_2][14];
		b_15=shared_1[it_2][15];
		sum_14+=(a*b_14);
		sum_15+=(a*b_15);
	}
	C(((idy*16)+14), idx)=sum_14;
	__syncthreads();
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(0-15)))+h), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(0-15)))+h), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_1[(tidx+0)][15]=B(0, (0+tidx));
		}
	}
	__syncthreads();
	#pragma unroll 
	for (it_2=0; it_2<16; it_2=(it_2+1))
	{
		float a;
		float b_15;
		a=shared_0[((tidx+(( - 1)*(it_2+0)))+16)];
		b_15=shared_1[it_2][15];
		sum_15+=(a*b_15);
	}
	C(((idy*16)+15), idx)=sum_15;
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
	{
		
	}
	{
		
	}
	{
		
	}
	{
		
	}
	{
		
	}
	{
		
	}
	{
		
	}
	{
		
	}
	{
		
	}
	{
		
	}
	{
		
	}
	{
		
	}
}
