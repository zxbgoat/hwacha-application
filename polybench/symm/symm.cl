/* OpenCL version of the PolyBenchC-4.2.1 kernel, written for hwacha-cc (this suite has no OpenCL
 * version of this case). The formulas and the order of floating-point operations follow kernel_*
 * in PolyBenchC-4.2.1 so the result matches the C reference exactly where the notes below say so. */
typedef float DATA_TYPE;

/* symm: C := alpha*A*B + beta*C, A symmetric (lower triangle stored). One work-item per C[i][j]:
 * the same element-wise sum the 4.2.1 loop nest produces, in the same order (temp2 over k < i,
 * then beta*C + alpha*B*A[i][i] + alpha*temp2, then the contributions of the rows below i). */
__kernel void symm_kernel(__global DATA_TYPE *C, __global DATA_TYPE *A, __global DATA_TYPE *B, DATA_TYPE alpha, DATA_TYPE beta, int m, int n)
{
	int j = get_global_id(0);
	int i = get_global_id(1);
	if ((i < m) && (j < n))
	{
		DATA_TYPE temp2 = 0;
		int k;
		for (k = 0; k < i; k++)
			temp2 += B[k*n + j] * A[i*m + k];
		DATA_TYPE c = beta * C[i*n + j] + alpha * B[i*n + j] * A[i*m + i] + alpha * temp2;
		for (k = i + 1; k < m; k++)
			c += alpha * B[k*n + j] * A[k*m + i];
		C[i*n + j] = c;
	}
}
