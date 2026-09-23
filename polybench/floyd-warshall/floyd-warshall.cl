/* OpenCL version of the PolyBenchC-4.2.1 kernel, written for hwacha-cc (this suite has no OpenCL
 * version of this case). Integer data, as 4.2.1. */
typedef int DATA_TYPE;
/* floyd-warshall: all-pairs shortest paths, k on the host, one work-item per (i, j). Row k and
 * column k do not change during step k, so the in-place update is race-free. */
__kernel void floyd_warshall_kernel(__global DATA_TYPE *path, int k, int n)
{
	int j = get_global_id(0);
	int i = get_global_id(1);
	if ((i < n) && (j < n))
	{
		DATA_TYPE s = path[i*n + k] + path[k*n + j];
		path[i*n + j] = path[i*n + j] < s ? path[i*n + j] : s;
	}
}
