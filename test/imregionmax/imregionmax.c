#define WIDTH_A (2048+16)
#define WIDTH_C 2048
#define COALESCED_NUM  16
#define A(y,x) A[(y)*WIDTH_A+(x)]
#define C(y,x) C[(y)*WIDTH_C+(x)]
__global__ void imregionmax(float* A, float* C, int width)
{
	float temp[9];
    int t;
    int i;
    int j;
    t = 0;
    for(i=0; i<3; i=i+1) {
		for(j=0; j<3; j=j+1){
			float a;
			a = A((idy+16-i), (idx+16-j));
			temp[t] = a;
			t=t+1;
		}
    }

    C(idy, idx) = cal(temp);
}


