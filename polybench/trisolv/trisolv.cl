/* OpenCL version of the PolyBenchC-4.2.1 kernel, written for hwacha-cc (this suite has no OpenCL
 * version of this case). The formulas and the order of floating-point operations follow kernel_*
 * in PolyBenchC-4.2.1 so the result matches the C reference exactly where the notes below say so. */
typedef float DATA_TYPE;

/* trisolv: forward substitution L x = b, column-oriented with i on the host: the work-item for
 * row i finishes x[i], the others subtract column i from the rows below. For every row the
 * subtractions happen in increasing column order, as 4.2.1's inner loop, so the result is exact. */
__kernel void trisolv_kernel(__global DATA_TYPE *L, __global DATA_TYPE *x, __global DATA_TYPE *b, int i, int n)
{
	int j = get_global_id(0) + i;
	if (j == i)
		x[i] = b[i] / L[i*n + i];
	else if (j < n)
		b[j] -= L[j*n + i] * (b[i] / L[i*n + i]);
}
