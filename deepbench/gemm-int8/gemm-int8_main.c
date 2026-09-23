// DeepBench inference GEMM, int8 inputs / int32 accumulate (gemm_bench inference int8:
// cublasGemmEx CUDA_R_8I x CUDA_R_8I -> CUDA_R_32I, NN shapes of the inference server / device sets,
// scaled down). Kernel: gemm-int8.cl (gemm_i8); the reference is exact integer arithmetic.
#include "common.h"
#define MAXM 128
#define MAXN 128
#define MAXK 128
#define LX 8
#define LY 8
void gemm_i8_ct(long n, int8_t *A, int8_t *B, int *C, int alpha, int beta, int m, int n_, int k, int lda, int ldb, int ldc);
static int8_t A[MAXM*MAXK], B[MAXK*MAXN];   // int8_t: on RISC-V plain char is unsigned, the OpenCL char is signed
static int C[MAXM*MAXN], ref[MAXM*MAXN];
static struct { int m, n, k; const char *from; } shapes[] = {
  {80, 44, 128, "inference server 5124x700x2048 /64,/16"}, {35, 44, 128, "inference server 35x700x2048 /16"},
  {120, 1, 80, "inference server 7680x1x2560 /64,/32"}, {96, 1, 80, "inference device 3072x1x1024 /32"},
  {64, 1, 76, "inference device 64x1x1216 /16"}, {96, 47, 128, "inference device 3072x1500x128 /32"} };
static int run(int s) {
  int m = shapes[s].m, n = shapes[s].n, k = shapes[s].k;
  for (int i = 0; i < m*k; i++) A[i] = (int8_t)(rnd() % 256);
  for (int i = 0; i < k*n; i++) B[i] = (int8_t)(rnd() % 256);
  for (int i = 0; i < m*n; i++) C[i] = ref[i] = 0;
  unsigned long c0 = cyc();
  for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) { int acc = 0; for (int l = 0; l < k; l++) acc += (int)A[i + l*m] * (int)B[l + j*k]; ref[i + j*m] = acc; }
  unsigned long c1 = cyc(); unsigned long sc = c1 - c0;
  c0 = cyc(); gemm_i8_ct(NDRANGE2(CEILDIV(m, LX), CEILDIV(n, LY), LX, LY), A, B, C, 1, 0, m, n, k, m, k, m); c1 = cyc();
  int bad = 0; for (int i = 0; i < m*n; i++) if (C[i] != ref[i]) bad++;
  printf("gemm-int8 %dx%dx%d (%s): scalar %lu cycles, hwacha-cc %lu cycles, %d mismatches / %d%s\n", m, n, k, shapes[s].from, sc, c1 - c0, bad, m*n, hwacha_vl_short ? " (VL SHORT)" : "");
  return bad;
}
int main(void) {
  int bad = 0; for (int s = 0; s < (int)(sizeof shapes / sizeof shapes[0]); s++) bad += run(s);
  printf("gemm-int8 %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
