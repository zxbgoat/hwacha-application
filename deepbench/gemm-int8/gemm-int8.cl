/* DeepBench inference GEMM in int8: 8-bit inputs, 32-bit accumulation (gemm_bench with the int8
 * precision calls cublasGemmEx with CUDA_R_8I inputs and CUDA_R_32I output, NN shapes). Column-major
 * as cuBLAS; one work-item per C element. Written for hwacha-cc. */
__kernel void gemm_i8(__global char *A, __global char *B, __global int *C, int alpha, int beta, int m, int n, int k, int lda, int ldb, int ldc)
{
	int i = get_global_id(0);
	int j = get_global_id(1);
	if ((i < m) && (j < n))
	{
		int acc = 0;
		int l;
		for (l = 0; l < k; l++)
			acc += (int)A[i + l*lda] * (int)B[l + j*ldb];
		C[i + j*ldc] = alpha * acc + beta * C[i + j*ldc];
	}
}
