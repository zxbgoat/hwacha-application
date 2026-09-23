/* DeepBench GEMM: C = alpha * op(A) * op(B) + beta * C, single precision, column-major as cuBLAS
 * (DeepBench's gemm_bench calls cublasSgemm with the transpose flags of its problem set: NN, TN and
 * NT shapes). op(A) is m x k, op(B) k x n, C m x n; lda / ldb / ldc as cuBLAS. One work-item per
 * C element, the k loop inside. Written for hwacha-cc. */
typedef float DATA_TYPE;
__kernel void gemm_nn(__global DATA_TYPE *A, __global DATA_TYPE *B, __global DATA_TYPE *C, DATA_TYPE alpha, DATA_TYPE beta, int m, int n, int k, int lda, int ldb, int ldc)
{
	int i = get_global_id(0);   /* row of C */
	int j = get_global_id(1);   /* column of C */
	if ((i < m) && (j < n))
	{
		DATA_TYPE acc = 0;
		int l;
		for (l = 0; l < k; l++)
			acc += A[i + l*lda] * B[l + j*ldb];
		C[i + j*ldc] = alpha * acc + beta * C[i + j*ldc];
	}
}
__kernel void gemm_tn(__global DATA_TYPE *A, __global DATA_TYPE *B, __global DATA_TYPE *C, DATA_TYPE alpha, DATA_TYPE beta, int m, int n, int k, int lda, int ldb, int ldc)
{
	int i = get_global_id(0);
	int j = get_global_id(1);
	if ((i < m) && (j < n))
	{
		DATA_TYPE acc = 0;
		int l;
		for (l = 0; l < k; l++)
			acc += A[l + i*lda] * B[l + j*ldb];   /* A stored k x m: op(A) = A^T */
		C[i + j*ldc] = alpha * acc + beta * C[i + j*ldc];
	}
}
__kernel void gemm_nt(__global DATA_TYPE *A, __global DATA_TYPE *B, __global DATA_TYPE *C, DATA_TYPE alpha, DATA_TYPE beta, int m, int n, int k, int lda, int ldb, int ldc)
{
	int i = get_global_id(0);
	int j = get_global_id(1);
	if ((i < m) && (j < n))
	{
		DATA_TYPE acc = 0;
		int l;
		for (l = 0; l < k; l++)
			acc += A[i + l*lda] * B[j + l*ldb];   /* B stored n x k: op(B) = B^T */
		C[i + j*ldc] = alpha * acc + beta * C[i + j*ldc];
	}
}
