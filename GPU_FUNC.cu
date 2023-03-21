#include <stdio.h>

/* ********************************************************* */
/* These two lines define the dimensions (MxN) of the matrix */
// #define M 4 // Number of elements/vector
// #define N 3 // Number of vectors
/* Change them to test different size matrices               */
/* ********************************************************* */

/* GPU Functions */
__global__ void printVectorKernel(float *v, int32_t M, int32_t N);
__global__ void printMatrixKernel(float *V, int32_t M, int32_t N);
__global__ void getVectorKernel(float *v, float *V_t, int rowNum, bool reverse, int32_t M, int32_t N);
__global__ void matrixTransposeKernel(float *V, float *V_t, bool reverse, int32_t M, int32_t N);

__global__ void matrixMultGPU(float *Q_t, float *A, float *R, int32_t M, int32_t N);

__global__ void calculateProjectionGPU(float *u, float *upper, float *lower, float *p, int32_t M, int32_t N);
__global__ void innerProductGPU(float *a, float *b, float *c, int32_t M, int32_t N);
__global__ void sumProjectionsGPU(float *P_t, float *projSum, int32_t M, int32_t N);
__global__ void vectorSubGPU(float *v, float *projSum, float *u, int32_t M, int32_t N);
__global__ void vectorNormsGPU(float *U_t, float *norms, int32_t M, int32_t N);
__global__ void normMultGPU(float *U, float *norms, float *E, int32_t M, int32_t N);

/* GPU Functions: */
// Prints a vector from the GPU
__global__ void printVectorKernel(float *v, int32_t M, int32_t N)
{
	if (threadIdx.x == 0)
	{
		for (int i = 0; i < M; i++)
		{
			printf("%f\t", v[i]);
		}
		printf("\n");
	}
}

// Prints a matrix from the GPU
__global__ void printMatrixKernel(float *V, int32_t M, int32_t N)
{
	if (threadIdx.x == 0)
	{
		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < N; j++)
			{
				printf("%f\t", V[i * N + j]);
			}
			printf("\n");
		}
		printf("\n");
	}
}

// Prints a matrix from the GPU
__global__ void printMatTKernel(float *V, int32_t M, int32_t N)
{
	if (threadIdx.x == 0)
	{
		for (int j = 0; j < N; j++)
		{
			for (int i = 0; i < M; i++)
			{
				printf("%f\t", V[j * M + i]);
			}
			printf("\n");
		}
		printf("\n");
	}
}

__global__ void printMatFlatKernel(float *V, int32_t M, int32_t N)
{
	if (threadIdx.x == 0)
	{
		for (int i = 0; i < N*M; i++)
		{
			printf("%f\t", V[i]);
		}
		printf("\n");
	}
}

// Transposes or reverse-transposes a matrix from GPU
__global__ void matrixTransposeKernel(float *V, float *V_t, bool reverse, int32_t M, int32_t N)
{
	if (threadIdx.x == 0)
	{
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < M; j++)
			{
				if (!reverse)
				{
					V_t[i * M + j] = V[j * N + i];
				}
				else
				{
					V[j * N + i] = V_t[i * M + j];
				}
			}
		}
	}
}

// Accesses a row in V_transpose and copies it into the storage vector v or does the reverse
__global__ void getVectorKernel(float *v, float *V_t, int rowNum, bool reverse, int32_t M, int32_t N)
{

	if (threadIdx.x == 0)
	{
		for (int i = 0; i < M; i++)
		{
			if (!reverse)
			{
				v[i] = V_t[rowNum * M + i];
			}
			else
			{
				V_t[rowNum * M + i] = v[i];
			}
		}
	}
}

// Multiply a vector by a scalar to get a projection - requires M threads for M-length vectors
__global__ void calculateProjectionGPU(float *u, float *upper, float *lower, float *p, int32_t M, int32_t N)
{

	int i = threadIdx.x;
	// Each thread does one multiplication
	if (i < M)
	{
		if (*lower != 0)
		{
			extern __shared__ float temp[];
			temp[i] = *upper / *lower;
			__syncthreads();
			p[i] = u[i] * temp[i];
		}
		else
		{
			p[i] = 0.0f;
		}
	}
}

// Calculate inner product on GPU - basically stolen from https://www.nvidia.com/content/GTC-2010/pdfs/2131_GTC2010.pdf
__global__ void innerProductGPU(float *a, float *b, float *c, int32_t M, int32_t N)
{

	// Likely to have more threads than entires, so use this to keep in range
	if (threadIdx.x < M)
	{
		// Each thread does one multiplication
		// Need to use shared memory to store products
		extern __shared__ float temp[];
		temp[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];
		// Need threads to synchronize - no threads advance until all are at this line, ensures no read-before-write hazard
		__syncthreads();
		// Now do the sum using only thread 0
		if (threadIdx.x == 0)
		{
			float sum = 0.0f;
			for (int i = 0; i < M; i++)
			{
				sum += temp[i];
			}
			*c = sum;
		}
	}
}

// Adds all of the projections onto one u of each v and returns that vector - requires M threads for M-length vectors
__global__ void sumProjectionsGPU(float *P_t, float *projSum, int32_t M, int32_t N)
{

	int idx = threadIdx.x;
	if (idx < M)
	{
		float temp = 0.0f;
		for (int i = 0; i < N; i++)
		{
			temp += P_t[i * M + idx];
		}
		projSum[idx] = temp;
	}
}

// Vector subtraction to get u[i] - requires M threads for M-length vectors, will be executed from 1 thread
__global__ void vectorSubGPU(float *v, float *projSum, float *u, int32_t M, int32_t N)
{

	int i = threadIdx.x;
	// Each thread subtracts one element from the other
	if (i < M)
	{
		u[i] = v[i] - projSum[i];
	}
}

// Calculates the eculidean norms of each vector and stores them into array - requires N threads for N columns
__global__ void vectorNormsGPU(float *U_t, float *norms, int32_t M, int32_t N)
{

	int idx = threadIdx.x;
	if (idx < N)
	{
		float temp = 0.0f;
		// First sum the components of each u together
		for (int i = 0; i < M; i++)
		{
			temp += (U_t[idx * M + i] * U_t[idx * M + i]);
		}
		// Now get reciprocal sqrt and store into norms array
		norms[idx] = rsqrtf(temp);
	}
}

// Mulitiplies each u by 1/norm to get the e's - requires M*N threads to do all at once
__global__ void normMultGPU(float *U, float *norms, float *E, int32_t M, int32_t N)
{

	// Note: This function requires that U be passed in, not U_t (for indexing purposes)
	int idx = threadIdx.x;
	if (idx < M * N)
	{
		// Get index in norms array
		int normIdx = (idx % N);
		E[idx] = U[idx] * norms[normIdx];
	}
}

__global__ void matrixMultGPU(float *Q_t, float *A, float *R, int32_t M, int32_t N)
{

	// Get each thread x and y
	int row = threadIdx.y;
	int col = threadIdx.x;

	// Q_t is a NxM matrix, A is a MxN matrix
	// Therefore R will be NxN
	if ((row < N) && (col < N))
	{
		for (int i = 0; i < M; i++)
		{
			R[row * N + col] += Q_t[row * M + i] * A[i * N + col];
		}
	}
}