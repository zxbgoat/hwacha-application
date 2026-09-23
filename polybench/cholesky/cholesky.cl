/* OpenCL version of the PolyBenchC-4.2.1 kernel, written for hwacha-cc (this suite has no OpenCL
 * version of this case). The formulas and the order of floating-point operations follow kernel_*
 * in PolyBenchC-4.2.1 so the result matches the C reference exactly where the notes below say so. */
typedef float DATA_TYPE;

/* cholesky: in-place lower Cholesky factor. 4.2.1 is row-oriented (left-looking); this is the
 * right-looking form, k on the host: kernel1 scales column k below the diagonal by sqrt(A[k][k]),
 * kernel2 takes the square root of the diagonal element, kernel3 updates the trailing lower
 * triangle. Every element sees the same subtractions in the same k order, so the result is exact. */
__kernel void cholesky_kernel1(__global DATA_TYPE *A, int k, int n)
{
	int i = get_global_id(0) + (k + 1);
	if (i < n)
		A[i*n + k] = A[i*n + k] / sqrt(A[k*n + k]);
}
__kernel void cholesky_kernel2(__global DATA_TYPE *A, int k, int n)
{
	if (get_global_id(0) == 0)
		A[k*n + k] = sqrt(A[k*n + k]);
}
__kernel void cholesky_kernel3(__global DATA_TYPE *A, int k, int n)
{
	int j = get_global_id(0) + (k + 1);
	int i = get_global_id(1) + (k + 1);
	if ((i < n) && (j <= i))
		A[i*n + j] -= A[i*n + k] * A[j*n + k];
}
