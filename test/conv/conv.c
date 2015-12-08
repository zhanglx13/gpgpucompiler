#define WIDTH_A (2048+16)
#define WIDTH_B 16
#define WIDTH_C 2048

#define A(y,x) A[(y)*WIDTH_A+(x)]
#define B(y,x) B[(y)*WIDTH_B+(x)]
#define C(y,x) C[(y)*WIDTH_C+(x)]
__global__ void conv(float *A, float *B, float *C, int width, int height, int w, int h) {
	int i;
	int j;
	float sum = 0;
	for (j=0; j<h; j=j+1) {
		for (i=0; i<w; i=i+1) {
			float a;
			float b;
			a = A(idy-j+h, idx-i+w);
			b = B(j, i);
			sum += a*b;
		}
	}
	C(idy, idx) = sum;
}


