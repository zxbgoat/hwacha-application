/* DeepBench GEMM in half precision (gemm_bench with the "half" precision: cublasGemmEx with
 * CUDA_R_16F inputs and output, CUDA_R_32F compute -- DeepBench's "FP16 inputs / FP32 math" mode).
 * Column-major as cuBLAS. Hwacha loads the halves with vlxh, converts with vfcvt.s.h, accumulates in
 * single precision (vfmadd.s) and rounds the result once with vfcvt.h.s. gemm_nn_h is the same
 * product in pure half arithmetic (vfmadd.h: half x half + half, one rounding per step), the
 * "FP16 math" mode no DeepBench platform lists but Hwacha has. Written for hwacha-cc. */
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void gemm_nn(__global half *A, __global half *B, __global half *C, float alpha, float beta, int m, int n, int k, int lda, int ldb, int ldc)
{
	int i = get_global_id(0);
	int j = get_global_id(1);
	if ((i < m) && (j < n))
	{
		float acc = 0;
		int l;
		for (l = 0; l < k; l++)
			acc += (float)A[i + l*lda] * (float)B[l + j*ldb];
		C[i + j*ldc] = (half)(alpha * acc + beta * (float)C[i + j*ldc]);
	}
}
__kernel void gemm_tn(__global half *A, __global half *B, __global half *C, float alpha, float beta, int m, int n, int k, int lda, int ldb, int ldc)
{
	int i = get_global_id(0);
	int j = get_global_id(1);
	if ((i < m) && (j < n))
	{
		float acc = 0;
		int l;
		for (l = 0; l < k; l++)
			acc += (float)A[l + i*lda] * (float)B[l + j*ldb];
		C[i + j*ldc] = (half)(alpha * acc + beta * (float)C[i + j*ldc]);
	}
}
__kernel void gemm_nt(__global half *A, __global half *B, __global half *C, float alpha, float beta, int m, int n, int k, int lda, int ldb, int ldc)
{
	int i = get_global_id(0);
	int j = get_global_id(1);
	if ((i < m) && (j < n))
	{
		float acc = 0;
		int l;
		for (l = 0; l < k; l++)
			acc += (float)A[i + l*lda] * (float)B[j + l*ldb];
		C[i + j*ldc] = (half)(alpha * acc + beta * (float)C[i + j*ldc]);
	}
}
__kernel void gemm_nn_h(__global half *A, __global half *B, __global half *C, int m, int n, int k, int lda, int ldb, int ldc)
{
	int i = get_global_id(0);
	int j = get_global_id(1);
	if ((i < m) && (j < n))
	{
		half acc = 0;
		int l;
		for (l = 0; l < k; l++)
			acc += A[i + l*lda] * B[l + j*ldb];
		C[i + j*ldc] = acc;
	}
}
