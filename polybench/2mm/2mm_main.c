// PolyBench 2mm: D := alpha*A*B*C + beta*D via tmp = alpha*A*B. Kernels: the unmodified PolyBench/GPU
// 2mm.cl (one work-item per tmp / D element); inputs and reference from PolyBenchC-4.2.1.
#include "common.h"
#define NI 32
#define NJ 32
#define NK 32
#define NL 32
#define LX 8
#define LY 8
void mm2_kernel1_ct(long n, float *tmp, float *A, float *B, int ni, int nj, int nk, int nl, float alpha, float beta);
void mm2_kernel2_ct(long n, float *tmp, float *C, float *D, int ni, int nj, int nk, int nl, float alpha, float beta);
static float A[NI*NK], B[NK*NJ], C[NJ*NL], D[NI*NL], tmp[NI*NJ], ref[NI*NL], rtmp[NI*NJ];
static const float alpha = 1.5f, beta = 1.2f;
static void ref_kernel(void) {
  for (int i = 0; i < NI; i++) for (int j = 0; j < NJ; j++) { rtmp[i*NJ + j] = 0; for (int k = 0; k < NK; k++) rtmp[i*NJ + j] += alpha * A[i*NK + k] * B[k*NJ + j]; }
  for (int i = 0; i < NI; i++) for (int j = 0; j < NL; j++) { ref[i*NL + j] *= beta; for (int k = 0; k < NJ; k++) ref[i*NL + j] += rtmp[i*NJ + k] * C[k*NL + j]; }
}
int main(void) {
  for (int i = 0; i < NI; i++) for (int j = 0; j < NK; j++) A[i*NK + j] = (float)((i*j+1) % NI) / NI;
  for (int i = 0; i < NK; i++) for (int j = 0; j < NJ; j++) B[i*NJ + j] = (float)(i*(j+1) % NJ) / NJ;
  for (int i = 0; i < NJ; i++) for (int j = 0; j < NL; j++) C[i*NL + j] = (float)((i*(j+3)+1) % NL) / NL;
  for (int i = 0; i < NI; i++) for (int j = 0; j < NL; j++) D[i*NL + j] = ref[i*NL + j] = (float)(i*(j+2) % NK) / NK;
  unsigned long c0 = cyc(); ref_kernel(); unsigned long c1 = cyc();
  REPORT("2mm scalar", c0, c1, (long)NI*NJ*NK + (long)NI*NL*NJ);
  c0 = cyc();
  mm2_kernel1_ct(NDRANGE2(CEILDIV(NJ, LX), CEILDIV(NI, LY), LX, LY), tmp, A, B, NI, NJ, NK, NL, alpha, beta);
  mm2_kernel2_ct(NDRANGE2(CEILDIV(NL, LX), CEILDIV(NI, LY), LX, LY), tmp, C, D, NI, NJ, NK, NL, alpha, beta);
  c1 = cyc(); REPORT("2mm hwacha-cc", c0, c1, (long)NI*NJ*NK + (long)NI*NL*NJ);
  int bad = check_f("2mm", D, ref, NI*NL, 1e-4f);
  printf("2mm %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
