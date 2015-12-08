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
#define merger_y 64
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
	float temp_16[9];
	float temp_17[9];
	float temp_18[9];
	float temp_19[9];
	float temp_20[9];
	float temp_21[9];
	float temp_22[9];
	float temp_23[9];
	float temp_24[9];
	float temp_25[9];
	float temp_26[9];
	float temp_27[9];
	float temp_28[9];
	float temp_29[9];
	float temp_30[9];
	float temp_31[9];
	float temp_32[9];
	float temp_33[9];
	float temp_34[9];
	float temp_35[9];
	float temp_36[9];
	float temp_37[9];
	float temp_38[9];
	float temp_39[9];
	float temp_40[9];
	float temp_41[9];
	float temp_42[9];
	float temp_43[9];
	float temp_44[9];
	float temp_45[9];
	float temp_46[9];
	float temp_47[9];
	float temp_48[9];
	float temp_49[9];
	float temp_50[9];
	float temp_51[9];
	float temp_52[9];
	float temp_53[9];
	float temp_54[9];
	float temp_55[9];
	float temp_56[9];
	float temp_57[9];
	float temp_58[9];
	float temp_59[9];
	float temp_60[9];
	float temp_61[9];
	float temp_62[9];
	float temp_63[9];
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
	int t_16;
	int t_17;
	int t_18;
	int t_19;
	int t_20;
	int t_21;
	int t_22;
	int t_23;
	int t_24;
	int t_25;
	int t_26;
	int t_27;
	int t_28;
	int t_29;
	int t_30;
	int t_31;
	int t_32;
	int t_33;
	int t_34;
	int t_35;
	int t_36;
	int t_37;
	int t_38;
	int t_39;
	int t_40;
	int t_41;
	int t_42;
	int t_43;
	int t_44;
	int t_45;
	int t_46;
	int t_47;
	int t_48;
	int t_49;
	int t_50;
	int t_51;
	int t_52;
	int t_53;
	int t_54;
	int t_55;
	int t_56;
	int t_57;
	int t_58;
	int t_59;
	int t_60;
	int t_61;
	int t_62;
	int t_63;
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
	t_16=0;
	t_17=0;
	t_18=0;
	t_19=0;
	t_20=0;
	t_21=0;
	t_22=0;
	t_23=0;
	t_24=0;
	t_25=0;
	t_26=0;
	t_27=0;
	t_28=0;
	t_29=0;
	t_30=0;
	t_31=0;
	t_32=0;
	t_33=0;
	t_34=0;
	t_35=0;
	t_36=0;
	t_37=0;
	t_38=0;
	t_39=0;
	t_40=0;
	t_41=0;
	t_42=0;
	t_43=0;
	t_44=0;
	t_45=0;
	t_46=0;
	t_47=0;
	t_48=0;
	t_49=0;
	t_50=0;
	t_51=0;
	t_52=0;
	t_53=0;
	t_54=0;
	t_55=0;
	t_56=0;
	t_57=0;
	t_58=0;
	t_59=0;
	t_60=0;
	t_61=0;
	t_62=0;
	t_63=0;
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(3-1)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(3-1)))+16), ((idx+(( - 1)*0))+16));
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
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(3-2)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(3-2)))+16), ((idx+(( - 1)*0))+16));
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
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(3-3)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(3-3)))+16), ((idx+(( - 1)*0))+16));
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
	C(((idy*64)+0), idx)=cal(temp_0);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-1)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-1)))+16), ((idx+(( - 1)*0))+16));
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
	C(((idy*64)+1), idx)=cal(temp_1);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-2)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-2)))+16), ((idx+(( - 1)*0))+16));
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
	C(((idy*64)+2), idx)=cal(temp_2);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-3)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-3)))+16), ((idx+(( - 1)*0))+16));
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
	C(((idy*64)+3), idx)=cal(temp_3);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-4)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-4)))+16), ((idx+(( - 1)*0))+16));
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
	C(((idy*64)+4), idx)=cal(temp_4);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-5)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-5)))+16), ((idx+(( - 1)*0))+16));
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
	C(((idy*64)+5), idx)=cal(temp_5);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-6)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-6)))+16), ((idx+(( - 1)*0))+16));
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
	C(((idy*64)+6), idx)=cal(temp_6);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-7)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-7)))+16), ((idx+(( - 1)*0))+16));
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
	C(((idy*64)+7), idx)=cal(temp_7);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-8)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-8)))+16), ((idx+(( - 1)*0))+16));
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
	C(((idy*64)+8), idx)=cal(temp_8);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-9)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-9)))+16), ((idx+(( - 1)*0))+16));
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
	C(((idy*64)+9), idx)=cal(temp_9);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-10)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-10)))+16), ((idx+(( - 1)*0))+16));
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
	C(((idy*64)+10), idx)=cal(temp_10);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-11)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-11)))+16), ((idx+(( - 1)*0))+16));
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
	C(((idy*64)+11), idx)=cal(temp_11);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-12)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-12)))+16), ((idx+(( - 1)*0))+16));
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
	C(((idy*64)+12), idx)=cal(temp_12);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-13)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-13)))+16), ((idx+(( - 1)*0))+16));
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
	C(((idy*64)+13), idx)=cal(temp_13);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-14)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-14)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_14[t_14]=a;
		temp_15[t_15]=a;
		temp_16[t_16]=a;
		t_14=(t_14+1);
		t_15=(t_15+1);
		t_16=(t_16+1);
	}
	C(((idy*64)+14), idx)=cal(temp_14);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-15)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-15)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_15[t_15]=a;
		temp_16[t_16]=a;
		temp_17[t_17]=a;
		t_15=(t_15+1);
		t_16=(t_16+1);
		t_17=(t_17+1);
	}
	C(((idy*64)+15), idx)=cal(temp_15);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-16)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-16)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_16[t_16]=a;
		temp_17[t_17]=a;
		temp_18[t_18]=a;
		t_16=(t_16+1);
		t_17=(t_17+1);
		t_18=(t_18+1);
	}
	C(((idy*64)+16), idx)=cal(temp_16);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-17)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-17)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_17[t_17]=a;
		temp_18[t_18]=a;
		temp_19[t_19]=a;
		t_17=(t_17+1);
		t_18=(t_18+1);
		t_19=(t_19+1);
	}
	C(((idy*64)+17), idx)=cal(temp_17);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-18)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-18)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_18[t_18]=a;
		temp_19[t_19]=a;
		temp_20[t_20]=a;
		t_18=(t_18+1);
		t_19=(t_19+1);
		t_20=(t_20+1);
	}
	C(((idy*64)+18), idx)=cal(temp_18);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-19)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-19)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_19[t_19]=a;
		temp_20[t_20]=a;
		temp_21[t_21]=a;
		t_19=(t_19+1);
		t_20=(t_20+1);
		t_21=(t_21+1);
	}
	C(((idy*64)+19), idx)=cal(temp_19);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-20)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-20)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_20[t_20]=a;
		temp_21[t_21]=a;
		temp_22[t_22]=a;
		t_20=(t_20+1);
		t_21=(t_21+1);
		t_22=(t_22+1);
	}
	C(((idy*64)+20), idx)=cal(temp_20);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-21)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-21)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_21[t_21]=a;
		temp_22[t_22]=a;
		temp_23[t_23]=a;
		t_21=(t_21+1);
		t_22=(t_22+1);
		t_23=(t_23+1);
	}
	C(((idy*64)+21), idx)=cal(temp_21);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-22)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-22)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_22[t_22]=a;
		temp_23[t_23]=a;
		temp_24[t_24]=a;
		t_22=(t_22+1);
		t_23=(t_23+1);
		t_24=(t_24+1);
	}
	C(((idy*64)+22), idx)=cal(temp_22);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-23)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-23)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_23[t_23]=a;
		temp_24[t_24]=a;
		temp_25[t_25]=a;
		t_23=(t_23+1);
		t_24=(t_24+1);
		t_25=(t_25+1);
	}
	C(((idy*64)+23), idx)=cal(temp_23);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-24)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-24)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_24[t_24]=a;
		temp_25[t_25]=a;
		temp_26[t_26]=a;
		t_24=(t_24+1);
		t_25=(t_25+1);
		t_26=(t_26+1);
	}
	C(((idy*64)+24), idx)=cal(temp_24);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-25)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-25)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_25[t_25]=a;
		temp_26[t_26]=a;
		temp_27[t_27]=a;
		t_25=(t_25+1);
		t_26=(t_26+1);
		t_27=(t_27+1);
	}
	C(((idy*64)+25), idx)=cal(temp_25);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-26)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-26)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_26[t_26]=a;
		temp_27[t_27]=a;
		temp_28[t_28]=a;
		t_26=(t_26+1);
		t_27=(t_27+1);
		t_28=(t_28+1);
	}
	C(((idy*64)+26), idx)=cal(temp_26);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-27)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-27)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_27[t_27]=a;
		temp_28[t_28]=a;
		temp_29[t_29]=a;
		t_27=(t_27+1);
		t_28=(t_28+1);
		t_29=(t_29+1);
	}
	C(((idy*64)+27), idx)=cal(temp_27);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-28)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-28)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_28[t_28]=a;
		temp_29[t_29]=a;
		temp_30[t_30]=a;
		t_28=(t_28+1);
		t_29=(t_29+1);
		t_30=(t_30+1);
	}
	C(((idy*64)+28), idx)=cal(temp_28);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-29)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-29)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_29[t_29]=a;
		temp_30[t_30]=a;
		temp_31[t_31]=a;
		t_29=(t_29+1);
		t_30=(t_30+1);
		t_31=(t_31+1);
	}
	C(((idy*64)+29), idx)=cal(temp_29);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-30)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-30)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_30[t_30]=a;
		temp_31[t_31]=a;
		temp_32[t_32]=a;
		t_30=(t_30+1);
		t_31=(t_31+1);
		t_32=(t_32+1);
	}
	C(((idy*64)+30), idx)=cal(temp_30);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-31)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-31)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_31[t_31]=a;
		temp_32[t_32]=a;
		temp_33[t_33]=a;
		t_31=(t_31+1);
		t_32=(t_32+1);
		t_33=(t_33+1);
	}
	C(((idy*64)+31), idx)=cal(temp_31);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-32)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-32)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_32[t_32]=a;
		temp_33[t_33]=a;
		temp_34[t_34]=a;
		t_32=(t_32+1);
		t_33=(t_33+1);
		t_34=(t_34+1);
	}
	C(((idy*64)+32), idx)=cal(temp_32);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-33)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-33)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_33[t_33]=a;
		temp_34[t_34]=a;
		temp_35[t_35]=a;
		t_33=(t_33+1);
		t_34=(t_34+1);
		t_35=(t_35+1);
	}
	C(((idy*64)+33), idx)=cal(temp_33);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-34)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-34)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_34[t_34]=a;
		temp_35[t_35]=a;
		temp_36[t_36]=a;
		t_34=(t_34+1);
		t_35=(t_35+1);
		t_36=(t_36+1);
	}
	C(((idy*64)+34), idx)=cal(temp_34);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-35)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-35)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_35[t_35]=a;
		temp_36[t_36]=a;
		temp_37[t_37]=a;
		t_35=(t_35+1);
		t_36=(t_36+1);
		t_37=(t_37+1);
	}
	C(((idy*64)+35), idx)=cal(temp_35);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-36)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-36)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_36[t_36]=a;
		temp_37[t_37]=a;
		temp_38[t_38]=a;
		t_36=(t_36+1);
		t_37=(t_37+1);
		t_38=(t_38+1);
	}
	C(((idy*64)+36), idx)=cal(temp_36);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-37)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-37)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_37[t_37]=a;
		temp_38[t_38]=a;
		temp_39[t_39]=a;
		t_37=(t_37+1);
		t_38=(t_38+1);
		t_39=(t_39+1);
	}
	C(((idy*64)+37), idx)=cal(temp_37);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-38)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-38)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_38[t_38]=a;
		temp_39[t_39]=a;
		temp_40[t_40]=a;
		t_38=(t_38+1);
		t_39=(t_39+1);
		t_40=(t_40+1);
	}
	C(((idy*64)+38), idx)=cal(temp_38);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-39)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-39)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_39[t_39]=a;
		temp_40[t_40]=a;
		temp_41[t_41]=a;
		t_39=(t_39+1);
		t_40=(t_40+1);
		t_41=(t_41+1);
	}
	C(((idy*64)+39), idx)=cal(temp_39);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-40)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-40)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_40[t_40]=a;
		temp_41[t_41]=a;
		temp_42[t_42]=a;
		t_40=(t_40+1);
		t_41=(t_41+1);
		t_42=(t_42+1);
	}
	C(((idy*64)+40), idx)=cal(temp_40);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-41)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-41)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_41[t_41]=a;
		temp_42[t_42]=a;
		temp_43[t_43]=a;
		t_41=(t_41+1);
		t_42=(t_42+1);
		t_43=(t_43+1);
	}
	C(((idy*64)+41), idx)=cal(temp_41);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-42)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-42)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_42[t_42]=a;
		temp_43[t_43]=a;
		temp_44[t_44]=a;
		t_42=(t_42+1);
		t_43=(t_43+1);
		t_44=(t_44+1);
	}
	C(((idy*64)+42), idx)=cal(temp_42);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-43)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-43)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_43[t_43]=a;
		temp_44[t_44]=a;
		temp_45[t_45]=a;
		t_43=(t_43+1);
		t_44=(t_44+1);
		t_45=(t_45+1);
	}
	C(((idy*64)+43), idx)=cal(temp_43);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-44)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-44)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_44[t_44]=a;
		temp_45[t_45]=a;
		temp_46[t_46]=a;
		t_44=(t_44+1);
		t_45=(t_45+1);
		t_46=(t_46+1);
	}
	C(((idy*64)+44), idx)=cal(temp_44);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-45)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-45)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_45[t_45]=a;
		temp_46[t_46]=a;
		temp_47[t_47]=a;
		t_45=(t_45+1);
		t_46=(t_46+1);
		t_47=(t_47+1);
	}
	C(((idy*64)+45), idx)=cal(temp_45);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-46)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-46)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_46[t_46]=a;
		temp_47[t_47]=a;
		temp_48[t_48]=a;
		t_46=(t_46+1);
		t_47=(t_47+1);
		t_48=(t_48+1);
	}
	C(((idy*64)+46), idx)=cal(temp_46);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-47)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-47)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_47[t_47]=a;
		temp_48[t_48]=a;
		temp_49[t_49]=a;
		t_47=(t_47+1);
		t_48=(t_48+1);
		t_49=(t_49+1);
	}
	C(((idy*64)+47), idx)=cal(temp_47);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-48)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-48)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_48[t_48]=a;
		temp_49[t_49]=a;
		temp_50[t_50]=a;
		t_48=(t_48+1);
		t_49=(t_49+1);
		t_50=(t_50+1);
	}
	C(((idy*64)+48), idx)=cal(temp_48);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-49)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-49)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_49[t_49]=a;
		temp_50[t_50]=a;
		temp_51[t_51]=a;
		t_49=(t_49+1);
		t_50=(t_50+1);
		t_51=(t_51+1);
	}
	C(((idy*64)+49), idx)=cal(temp_49);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-50)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-50)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_50[t_50]=a;
		temp_51[t_51]=a;
		temp_52[t_52]=a;
		t_50=(t_50+1);
		t_51=(t_51+1);
		t_52=(t_52+1);
	}
	C(((idy*64)+50), idx)=cal(temp_50);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-51)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-51)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_51[t_51]=a;
		temp_52[t_52]=a;
		temp_53[t_53]=a;
		t_51=(t_51+1);
		t_52=(t_52+1);
		t_53=(t_53+1);
	}
	C(((idy*64)+51), idx)=cal(temp_51);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-52)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-52)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_52[t_52]=a;
		temp_53[t_53]=a;
		temp_54[t_54]=a;
		t_52=(t_52+1);
		t_53=(t_53+1);
		t_54=(t_54+1);
	}
	C(((idy*64)+52), idx)=cal(temp_52);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-53)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-53)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_53[t_53]=a;
		temp_54[t_54]=a;
		temp_55[t_55]=a;
		t_53=(t_53+1);
		t_54=(t_54+1);
		t_55=(t_55+1);
	}
	C(((idy*64)+53), idx)=cal(temp_53);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-54)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-54)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_54[t_54]=a;
		temp_55[t_55]=a;
		temp_56[t_56]=a;
		t_54=(t_54+1);
		t_55=(t_55+1);
		t_56=(t_56+1);
	}
	C(((idy*64)+54), idx)=cal(temp_54);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-55)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-55)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_55[t_55]=a;
		temp_56[t_56]=a;
		temp_57[t_57]=a;
		t_55=(t_55+1);
		t_56=(t_56+1);
		t_57=(t_57+1);
	}
	C(((idy*64)+55), idx)=cal(temp_55);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-56)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-56)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_56[t_56]=a;
		temp_57[t_57]=a;
		temp_58[t_58]=a;
		t_56=(t_56+1);
		t_57=(t_57+1);
		t_58=(t_58+1);
	}
	C(((idy*64)+56), idx)=cal(temp_56);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-57)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-57)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_57[t_57]=a;
		temp_58[t_58]=a;
		temp_59[t_59]=a;
		t_57=(t_57+1);
		t_58=(t_58+1);
		t_59=(t_59+1);
	}
	C(((idy*64)+57), idx)=cal(temp_57);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-58)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-58)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_58[t_58]=a;
		temp_59[t_59]=a;
		temp_60[t_60]=a;
		t_58=(t_58+1);
		t_59=(t_59+1);
		t_60=(t_60+1);
	}
	C(((idy*64)+58), idx)=cal(temp_58);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-59)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-59)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_59[t_59]=a;
		temp_60[t_60]=a;
		temp_61[t_61]=a;
		t_59=(t_59+1);
		t_60=(t_60+1);
		t_61=(t_61+1);
	}
	C(((idy*64)+59), idx)=cal(temp_59);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-60)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-60)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_60[t_60]=a;
		temp_61[t_61]=a;
		temp_62[t_62]=a;
		t_60=(t_60+1);
		t_61=(t_61+1);
		t_62=(t_62+1);
	}
	C(((idy*64)+60), idx)=cal(temp_60);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-61)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-61)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_61[t_61]=a;
		temp_62[t_62]=a;
		temp_63[t_63]=a;
		t_61=(t_61+1);
		t_62=(t_62+1);
		t_63=(t_63+1);
	}
	C(((idy*64)+61), idx)=cal(temp_61);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-62)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-62)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_62[t_62]=a;
		temp_63[t_63]=a;
		t_62=(t_62+1);
		t_63=(t_63+1);
	}
	C(((idy*64)+62), idx)=cal(temp_62);
	__syncthreads();
	if ((tidx<16))
	{
		{
			shared_0[(tidx+0)]=A((((idy*64)+(( - 1)*(0-63)))+16), (idx+(( - 1)*0)));
		}
	}
	{
		shared_0[(tidx+16)]=A((((idy*64)+(( - 1)*(0-63)))+16), ((idx+(( - 1)*0))+16));
	}
	__syncthreads();
	#pragma unroll 
	for (it_1=0; it_1<3; it_1=(it_1+1))
	{
		float a;
		a=shared_0[((tidx+(( - 1)*it_1))+16)];
		temp_63[t_63]=a;
		t_63=(t_63+1);
	}
	C(((idy*64)+63), idx)=cal(temp_63);
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
