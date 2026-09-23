/* OpenCL version of the PolyBenchC-4.2.1 kernel, written for hwacha-cc (this suite has no OpenCL
 * version of this case). The formulas and the order of floating-point operations follow kernel_*
 * in PolyBenchC-4.2.1 so the result matches the C reference exactly where the notes below say so. */
typedef float DATA_TYPE;

/* doitgen: A[r][q][:] := A[r][q][:] * C4 for every (r, q). One work-item per (r, q); sum is the
 * per-(r, q) slice of a global scratch array, as in 4.2.1 (written, then copied back). */
__kernel void doitgen_kernel(__global DATA_TYPE *A, __global DATA_TYPE *C4, __global DATA_TYPE *sum, int nr, int nq, int np)
{
	int q = get_global_id(0);
	int r = get_global_id(1);
	if ((r < nr) && (q < nq))
	{
		int p, s;
		__global DATA_TYPE *a = A + (r*nq + q)*np;
		__global DATA_TYPE *su = sum + (r*nq + q)*np;
		for (p = 0; p < np; p++)
		{
			su[p] = 0;
			for (s = 0; s < np; s++)
				su[p] += a[s] * C4[s*np + p];
		}
		for (p = 0; p < np; p++)
			a[p] = su[p];
	}
}
