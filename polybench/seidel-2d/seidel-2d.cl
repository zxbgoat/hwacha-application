/* OpenCL version of the PolyBenchC-4.2.1 kernel, written for hwacha-cc (this suite has no OpenCL
 * version of this case). The formulas and the order of floating-point operations follow kernel_*
 * in PolyBenchC-4.2.1 so the result matches the C reference exactly where the notes below say so. */
typedef float DATA_TYPE;

/* seidel-2d: Gauss-Seidel 9-point sweep, updated in place, so within a sweep A[i][j] depends on
 * the already-updated A[i-1][*] and A[i][j-1]. The host walks the anti-diagonals d = i + j and one
 * work-item handles every interior (i, d - i) on the diagonal; every element sees exactly the
 * neighbour values the 4.2.1 order produces. */
__kernel void seidel_2d_kernel(__global DATA_TYPE *A, int d, int n)
{
	int i = get_global_id(0) + 1;
	int j = d - i;
	if ((i <= n-2) && (j >= 1) && (j <= n-2))
		A[i*n + j] = (A[(i-1)*n + (j-1)] + A[(i-1)*n + j] + A[(i-1)*n + (j+1)]
		            + A[i*n + (j-1)] + A[i*n + j] + A[i*n + (j+1)]
		            + A[(i+1)*n + (j-1)] + A[(i+1)*n + j] + A[(i+1)*n + (j+1)]) / 9.0f;
}
