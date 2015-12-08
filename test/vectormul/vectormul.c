#define WIDTH_C 2048
#define COALESCED_NUM  16
#define C(y,x) C[(y)*WIDTH_C+(x)]
__global__ void vectormul(float *A, float *B, float *C, int width) {

	float sum;
	float a;
	float b;
	sum = 0;
	a = A[idy];
	b = B[idx];
	sum += a*b;
	C(idy, idx)+=sum;
}


