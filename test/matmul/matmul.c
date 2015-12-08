#define WIDTH_A 2048
#define WIDTH_B 2048
#define WIDTH_C 2048
#define A(y,x) A[(y)*WIDTH_A+(x)]
#define B(y,x) B[(y)*WIDTH_B+(x)]
#define C(y,x) C[(y)*WIDTH_C+(x)]
__global__ void matmul(float *A, float *B, float *C, int width, int height) {
	int i;
	float sum;
	sum = 0;
	for (i=0; i<width; i=i+1) {
		float a;
		float b;
		a = A(idy, i);
		b = B(i, idx);
		sum += a*b;
	}
	C(idy, idx) = sum;
}

