// PolyBench gemm: C := alpha*A*B + beta*C, one work-item per C element.
// Kernel: the unmodified PolyBench/GPU gemm.cl; the reference is kernel_gemm from
// PolyBenchC-4.2.1 (same formula, element-at-a-time), the inputs its init_array.
#include "common.h"
#define NI 48
#define NJ 48
#define NK 48
#define LX 8
#define LY 8
void gemm_ct(long n, float *a, float *b, float *c, float alpha, float beta, int ni, int nj, int nk);
static float A[NI*NK], B[NK*NJ], C[NI*NJ], ref[NI*NJ];
static const float alpha = 1.5f, beta = 1.2f;   // PolyBenchC-4.2.1 init_array
static void ref_kernel(int ni, int nj, int nk) {
  for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nj; j++) ref[i*nj + j] *= beta;
    for (int k = 0; k < nk; k++) for (int j = 0; j < nj; j++) ref[i*nj + j] += alpha * A[i*nk + k] * B[k*nj + j];
  }
}
int main(void) {
  for (int i = 0; i < NI; i++) for (int j = 0; j < NJ; j++) { C[i*NJ + j] = (float)((i*j+1) % NI) / NI; ref[i*NJ + j] = C[i*NJ + j]; }
  for (int i = 0; i < NI; i++) for (int j = 0; j < NK; j++) A[i*NK + j] = (float)(i*(j+1) % NK) / NK;
  for (int i = 0; i < NK; i++) for (int j = 0; j < NJ; j++) B[i*NJ + j] = (float)(i*(j+2) % NJ) / NJ;
  unsigned long c0 = cyc(); ref_kernel(NI, NJ, NK); unsigned long c1 = cyc();
  REPORT("gemm scalar", c0, c1, (long)NI*NJ*NK);
  c0 = cyc(); gemm_ct(NDRANGE2(CEILDIV(NJ, LX), CEILDIV(NI, LY), LX, LY), A, B, C, alpha, beta, NI, NJ, NK); c1 = cyc();
  REPORT("gemm hwacha-cc", c0, c1, (long)NI*NJ*NK);
  int bad = check_f("gemm", C, ref, NI*NJ, 1e-4f);
  printf("gemm %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
