#define globalDimY 1
#define globalDimX 262144
__global__ void reduction(float* A, int size, int segSize) {
#pragma gCompiler gValue segSize 262144

	int i;
	int k;
	float sum;
	sum = 0;
	for (k=0; k<size; k=k+segSize) {
		float r;
		r = A[idx+k];
		sum += r;
	}
	A[idx] = sum;
	__syncthreads();
	for (i=1; i<segSize; i=i*2) {
		if (idx<segSize/i/2) {
			float a;
			float b;
			float c;
			a = A[idx];
			b = A[idx+segSize/i/2];
			c = a+b;
			A[idx] = c;
		}
		__syncthreads();
	}
}
