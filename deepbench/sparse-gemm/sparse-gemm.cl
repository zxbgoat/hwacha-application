/* DeepBench sparse GEMM: C = alpha * A_sparse * B + beta * C with A (m x k) in CSR, B (k x n) and C
 * (m x n) dense column-major, as cusparseScsrmm in sparse_bench (alpha = 1 / k, beta = 0). A's
 * sparsity (0.9 / 0.95 in the problem set) is produced on the host as DeepBench does: uniform random
 * values, those below the sparsity threshold zeroed. One work-item per C element, walking row i's
 * non-zeros. Written for hwacha-cc. */
typedef float DATA_TYPE;
__kernel void csrmm(__global DATA_TYPE *val, __global int *rowptr, __global int *colind, __global DATA_TYPE *B, __global DATA_TYPE *C, DATA_TYPE alpha, DATA_TYPE beta, int m, int n, int k)
{
	int i = get_global_id(0);
	int j = get_global_id(1);
	if ((i < m) && (j < n))
	{
		DATA_TYPE acc = 0;
		int p;
		for (p = rowptr[i]; p < rowptr[i+1]; p++)
			acc += val[p] * B[colind[p] + j*k];
		C[i + j*m] = alpha * acc + beta * C[i + j*m];
	}
}
