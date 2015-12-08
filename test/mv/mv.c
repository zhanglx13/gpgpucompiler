#define WIDTH_A 2048
#define COALESCED_NUM  16
#define A(y,x) A[(y)*WIDTH_A+(x)]
#define globalDimY 1
__global__ void mv(float *A, float *B, float *C, int width) {
	int i;
	float sum;
	sum = 0;

	for (i=0; i<WIDTH_A; i=i+1) {
		float a;
		float b;
		a = A(idx, i);
		b = B[i];
		sum += a*b;
	}
	C[idx] = sum;
}


