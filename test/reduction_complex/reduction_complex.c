#define globalDimY 1
#define globalDimX 262144
__global__ void reduction_complex(float* A, float* B, int size, int segSize) {
#pragma gCompiler gValue segSize 262144

	int i;
	int k;
	float sum;
	sum = 0;
	for (k=0; k<size; k=k+262144) {
		float real;
		float img;
		real = A[2*idx+2*k+1];
		img = A[2*idx+2*k];
		sum += real;
		sum += img;
	}
	B[idx] = sum;
	__syncthreads();
	for (i=1; i<segSize; i=i*2) {
		if (idx<segSize/i/2) {
			float a;
			float b;
			float c;
			a = B[idx];
			b = B[idx+segSize/i/2];
			c = a+b;
			B[idx] = c;
		}
		__syncthreads();
	}
}
