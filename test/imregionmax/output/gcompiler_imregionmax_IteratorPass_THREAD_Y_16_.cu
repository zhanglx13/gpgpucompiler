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
#define A(y,x) A[(y)*WIDTH_A+(x)]
__global__ void imregionmax(float * A, float * C, int width)
{
	__shared__ float shared_0[272];
	float temp_0[9];
	float temp_1[9];
	float temp_2[9];
	float temp_3[9];
	float temp_4[9];
	float temp_5[9];
	float temp_6[9];
	float temp_7[9];
	float temp_8[9];
	float temp_9[9];
	float temp_10[9];
	float temp_11[9];
	float temp_12[9];
	float temp_13[9];
	float temp_14[9];
	float temp_15[9];
	int t_0;
	int t_1;
	int t_2;
	int t_3;
	int t_4;
	int t_5;
	int t_6;
	int t_7;
	int t_8;
	int t_9;
	int t_10;
	int t_11;
	int t_12;
	int t_13;
	int t_14;
	int t_15;
	int it_1;
	t_0=0;
	t_1=0;
	t_2=0;
	t_3=0;
	t_4=0;
	t_5=0;
	t_6=0;
	t_7=0;
	t_8=0;
	t_9=0;
	t_10=0;
	t_11=0;
	t_12=0;
	t_13=0;
	t_14=0;
	t_15=0;
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(3-1)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(3-1)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_0[t_0]=a;
		t_0=(t_0+1);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(3-2)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(3-2)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_0[t_0]=a;
		temp_1[t_1]=a;
		t_0=(t_0+1);
		t_1=(t_1+1);
	}
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(3-3)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(3-3)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_0[t_0]=a;
		temp_1[t_1]=a;
		temp_2[t_2]=a;
		t_0=(t_0+1);
		t_1=(t_1+1);
		t_2=(t_2+1);
	}
	C(((idy*16)+0), idx)=cal(temp_0);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(0-1)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(0-1)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_1[t_1]=a;
		temp_2[t_2]=a;
		temp_3[t_3]=a;
		t_1=(t_1+1);
		t_2=(t_2+1);
		t_3=(t_3+1);
	}
	C(((idy*16)+1), idx)=cal(temp_1);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(0-2)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(0-2)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_2[t_2]=a;
		temp_3[t_3]=a;
		temp_4[t_4]=a;
		t_2=(t_2+1);
		t_3=(t_3+1);
		t_4=(t_4+1);
	}
	C(((idy*16)+2), idx)=cal(temp_2);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(0-3)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(0-3)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_3[t_3]=a;
		temp_4[t_4]=a;
		temp_5[t_5]=a;
		t_3=(t_3+1);
		t_4=(t_4+1);
		t_5=(t_5+1);
	}
	C(((idy*16)+3), idx)=cal(temp_3);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(0-4)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(0-4)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_4[t_4]=a;
		temp_5[t_5]=a;
		temp_6[t_6]=a;
		t_4=(t_4+1);
		t_5=(t_5+1);
		t_6=(t_6+1);
	}
	C(((idy*16)+4), idx)=cal(temp_4);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(0-5)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(0-5)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_5[t_5]=a;
		temp_6[t_6]=a;
		temp_7[t_7]=a;
		t_5=(t_5+1);
		t_6=(t_6+1);
		t_7=(t_7+1);
	}
	C(((idy*16)+5), idx)=cal(temp_5);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(0-6)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(0-6)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_6[t_6]=a;
		temp_7[t_7]=a;
		temp_8[t_8]=a;
		t_6=(t_6+1);
		t_7=(t_7+1);
		t_8=(t_8+1);
	}
	C(((idy*16)+6), idx)=cal(temp_6);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(0-7)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(0-7)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_7[t_7]=a;
		temp_8[t_8]=a;
		temp_9[t_9]=a;
		t_7=(t_7+1);
		t_8=(t_8+1);
		t_9=(t_9+1);
	}
	C(((idy*16)+7), idx)=cal(temp_7);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(0-8)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(0-8)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_8[t_8]=a;
		temp_9[t_9]=a;
		temp_10[t_10]=a;
		t_8=(t_8+1);
		t_9=(t_9+1);
		t_10=(t_10+1);
	}
	C(((idy*16)+8), idx)=cal(temp_8);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(0-9)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(0-9)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_9[t_9]=a;
		temp_10[t_10]=a;
		temp_11[t_11]=a;
		t_9=(t_9+1);
		t_10=(t_10+1);
		t_11=(t_11+1);
	}
	C(((idy*16)+9), idx)=cal(temp_9);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(0-10)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(0-10)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_10[t_10]=a;
		temp_11[t_11]=a;
		temp_12[t_12]=a;
		t_10=(t_10+1);
		t_11=(t_11+1);
		t_12=(t_12+1);
	}
	C(((idy*16)+10), idx)=cal(temp_10);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(0-11)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(0-11)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_11[t_11]=a;
		temp_12[t_12]=a;
		temp_13[t_13]=a;
		t_11=(t_11+1);
		t_12=(t_12+1);
		t_13=(t_13+1);
	}
	C(((idy*16)+11), idx)=cal(temp_11);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(0-12)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(0-12)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_12[t_12]=a;
		temp_13[t_13]=a;
		temp_14[t_14]=a;
		t_12=(t_12+1);
		t_13=(t_13+1);
		t_14=(t_14+1);
	}
	C(((idy*16)+12), idx)=cal(temp_12);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(0-13)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(0-13)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_13[t_13]=a;
		temp_14[t_14]=a;
		temp_15[t_15]=a;
		t_13=(t_13+1);
		t_14=(t_14+1);
		t_15=(t_15+1);
	}
	C(((idy*16)+13), idx)=cal(temp_13);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(0-14)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(0-14)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_14[t_14]=a;
		temp_15[t_15]=a;
		t_14=(t_14+1);
		t_15=(t_15+1);
	}
	C(((idy*16)+14), idx)=cal(temp_14);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*16)+(( - 1)*(0-15)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*16)+(( - 1)*(0-15)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_15[t_15]=a;
		t_15=(t_15+1);
	}
	C(((idy*16)+15), idx)=cal(temp_15);
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
