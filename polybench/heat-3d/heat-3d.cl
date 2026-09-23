/* OpenCL version of the PolyBenchC-4.2.1 kernel, written for hwacha-cc (this suite has no OpenCL
 * version of this case). The formulas and the order of floating-point operations follow kernel_*
 * in PolyBenchC-4.2.1 so the result matches the C reference exactly where the notes below say so. */
typedef float DATA_TYPE;

/* heat-3d: 7-point 3-D heat stencil, one work-item per (i, j) interior column with the k loop
 * inside; the host alternates the two half-steps (A -> B, B -> A) by swapping the arguments. */
__kernel void heat_3d_kernel(__global DATA_TYPE *A, __global DATA_TYPE *B, int n)
{
	int j = get_global_id(0) + 1;
	int i = get_global_id(1) + 1;
	if ((i < n-1) && (j < n-1))
	{
		int k;
		for (k = 1; k < n-1; k++)
		{
			B[(i*n + j)*n + k] = 0.125f * (A[((i+1)*n + j)*n + k] - 2.0f * A[(i*n + j)*n + k] + A[((i-1)*n + j)*n + k])
			                   + 0.125f * (A[(i*n + (j+1))*n + k] - 2.0f * A[(i*n + j)*n + k] + A[(i*n + (j-1))*n + k])
			                   + 0.125f * (A[(i*n + j)*n + (k+1)] - 2.0f * A[(i*n + j)*n + k] + A[(i*n + j)*n + (k-1)])
			                   + A[(i*n + j)*n + k];
		}
	}
}
