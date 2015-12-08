#define WIDTH_A 2048
#define WIDTH_C 2048
#define COALESCED_NUM  32
#define A(y,x) A[(y)*WIDTH_A+(x)]
#define C(y,x) C[(y)*WIDTH_C+(x)]
__global__ void transpose(float *A, float *C, int width) {
	int i = 0;
	float sum = 0;

	sum = A(idx, idy);
	C(idy, idx) = sum;
}


