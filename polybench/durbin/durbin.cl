/* OpenCL version of the PolyBenchC-4.2.1 kernel, written for hwacha-cc (this suite has no OpenCL
 * version of this case). The formulas and the order of floating-point operations follow kernel_*
 * in PolyBenchC-4.2.1 so the result matches the C reference exactly where the notes below say so. */
typedef float DATA_TYPE;

/* durbin: Levinson-Durbin recursion, k on the host. kernel1 (one work-item) advances beta, computes
 * sum and alpha for step k and keeps them in st[0] = alpha, st[1] = beta; kernel2 (k work-items)
 * forms z; kernel3 (k + 1 work-items) copies z back and sets y[k] = alpha. */
__kernel void durbin_kernel1(__global DATA_TYPE *r, __global DATA_TYPE *y, __global DATA_TYPE *st, int k, int n)
{
	if (get_global_id(0) == 0)
	{
		DATA_TYPE alpha = st[0], beta = st[1], sum = 0;
		int i;
		beta = (1 - alpha*alpha) * beta;
		for (i = 0; i < k; i++)
			sum += r[k-i-1] * y[i];
		alpha = -(r[k] + sum) / beta;
		st[0] = alpha; st[1] = beta;
	}
}
__kernel void durbin_kernel2(__global DATA_TYPE *y, __global DATA_TYPE *z, __global DATA_TYPE *st, int k)
{
	int i = get_global_id(0);
	if (i < k)
		z[i] = y[i] + st[0] * y[k-i-1];
}
__kernel void durbin_kernel3(__global DATA_TYPE *y, __global DATA_TYPE *z, __global DATA_TYPE *st, int k)
{
	int i = get_global_id(0);
	if (i < k)
		y[i] = z[i];
	else if (i == k)
		y[k] = st[0];
}
