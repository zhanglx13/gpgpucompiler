#define WIDTH_A 2048
#define COALESCED_NUM  32
#define globalDimY 1
#define A(y,x) A[(y)*WIDTH_A+(x)]
__global__ void tmv(float *A, float *B, float *C, int width) {
	int i;
	i = 0;
	float sum;
	sum = 0;

	for (i=0; i<width; i=i+1) {
		float a;
		float b;
		a = A(i, idx);
		b = B[i];
		sum += a*b;
	}
	C[idx] = sum;
}


