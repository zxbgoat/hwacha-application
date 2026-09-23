/* OpenCL version of the PolyBenchC-4.2.1 kernel, written for hwacha-cc (this suite has no OpenCL
 * version of this case). The formulas and the order of floating-point operations follow kernel_*
 * in PolyBenchC-4.2.1 so the result matches the C reference exactly where the notes below say so. */
typedef float DATA_TYPE;

/* ludcmp: LU (unit lower L, as 4.2.1's Doolittle form) then forward and back substitution.
 * The LU is the right-looking form with k on the host (kernel1 scales column k, kernel2 the
 * trailing update); every A[i][j] sees the same subtractions in the same k order as 4.2.1 and the
 * division of L entries happens after the same subtractions, so the factor is exact. The
 * substitutions are column-oriented, one launch per i: the forward one subtracts in the same j
 * order as 4.2.1 (exact); the backward one subtracts in the opposite order (rounding-level
 * differences, compared with a tolerance). */
__kernel void ludcmp_kernel1(__global DATA_TYPE *A, int k, int n)
{
	int i = get_global_id(0) + (k + 1);
	if (i < n)
		A[i*n + k] = A[i*n + k] / A[k*n + k];
}
__kernel void ludcmp_kernel2(__global DATA_TYPE *A, int k, int n)
{
	int j = get_global_id(0) + (k + 1);
	int i = get_global_id(1) + (k + 1);
	if ((i < n) && (j < n))
		A[i*n + j] -= A[i*n + k] * A[k*n + j];
}
/* forward: y[i] = w[i] (w = b minus what earlier columns subtracted), then w[j] -= A[j][i]*y[i], j > i */
__kernel void ludcmp_kernel3(__global DATA_TYPE *A, __global DATA_TYPE *w, __global DATA_TYPE *y, int i, int n)
{
	int j = get_global_id(0) + i;
	if (j == i)
		y[i] = w[i];
	else if (j < n)
		w[j] -= A[j*n + i] * w[i];
}
/* backward: x[i] = w[i] / A[i][i], then w[j] -= A[j][i]*x[i], j < i */
__kernel void ludcmp_kernel4(__global DATA_TYPE *A, __global DATA_TYPE *w, __global DATA_TYPE *x, int i, int n)
{
	int j = get_global_id(0);
	if (j == i)
		x[i] = w[i] / A[i*n + i];
	else if (j < i)
		w[j] -= A[j*n + i] * (w[i] / A[i*n + i]);
}
