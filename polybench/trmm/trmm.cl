/* OpenCL version of the PolyBenchC-4.2.1 kernel, written for hwacha-cc (this suite has no OpenCL
 * version of this case). The formulas and the order of floating-point operations follow kernel_*
 * in PolyBenchC-4.2.1 so the result matches the C reference exactly where the notes below say so. */
typedef float DATA_TYPE;

/* trmm: B := alpha*A^T*B, A unit lower triangular. One work-item per B[i][j]; the 4.2.1 loop reads
 * rows k > i of B before they are updated, so the kernel reads Bin and writes Bout. */
__kernel void trmm_kernel(__global DATA_TYPE *A, __global DATA_TYPE *Bin, __global DATA_TYPE *Bout, DATA_TYPE alpha, int m, int n)
{
	int j = get_global_id(0);
	int i = get_global_id(1);
	if ((i < m) && (j < n))
	{
		DATA_TYPE b = Bin[i*n + j];
		int k;
		for (k = i + 1; k < m; k++)
			b += A[k*m + i] * Bin[k*n + j];
		Bout[i*n + j] = alpha * b;
	}
}
