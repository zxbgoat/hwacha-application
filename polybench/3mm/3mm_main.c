// PolyBench 3mm: G := (A*B) * (C*D) via E = A*B and F = C*D. Kernels: the unmodified PolyBench/GPU
// 3mm.cl (three matrix products, one work-item per output element); inputs / reference from 4.2.1.
#include "common.h"
#define NI 32
#define NJ 32
#define NK 32
#define NL 32
#define NM 32
#define LX 8
#define LY 8
void mm3_kernel1_ct(long n, float *A, float *B, float *E, int ni, int nj, int nk);
void mm3_kernel2_ct(long n, float *C, float *D, float *F, int nj, int nl, int nm);
void mm3_kernel3_ct(long n, float *E, float *F, float *G, int ni, int nl, int nj);
static float A[NI*NK], B[NK*NJ], C[NJ*NM], D[NM*NL], E[NI*NJ], F[NJ*NL], G[NI*NL], rE[NI*NJ], rF[NJ*NL], ref[NI*NL];
static void ref_kernel(void) {
  for (int i = 0; i < NI; i++) for (int j = 0; j < NJ; j++) { rE[i*NJ + j] = 0; for (int k = 0; k < NK; k++) rE[i*NJ + j] += A[i*NK + k] * B[k*NJ + j]; }
  for (int i = 0; i < NJ; i++) for (int j = 0; j < NL; j++) { rF[i*NL + j] = 0; for (int k = 0; k < NM; k++) rF[i*NL + j] += C[i*NM + k] * D[k*NL + j]; }
  for (int i = 0; i < NI; i++) for (int j = 0; j < NL; j++) { ref[i*NL + j] = 0; for (int k = 0; k < NJ; k++) ref[i*NL + j] += rE[i*NJ + k] * rF[k*NL + j]; }
}
int main(void) {
  for (int i = 0; i < NI; i++) for (int j = 0; j < NK; j++) A[i*NK + j] = (float)((i*j+1) % NI) / (5*NI);
  for (int i = 0; i < NK; i++) for (int j = 0; j < NJ; j++) B[i*NJ + j] = (float)((i*(j+1)+2) % NJ) / (5*NJ);
  for (int i = 0; i < NJ; i++) for (int j = 0; j < NM; j++) C[i*NM + j] = (float)(i*(j+3) % NL) / (5*NL);
  for (int i = 0; i < NM; i++) for (int j = 0; j < NL; j++) D[i*NL + j] = (float)((i*(j+2)+2) % NK) / (5*NK);
  unsigned long c0 = cyc(); ref_kernel(); unsigned long c1 = cyc();
  REPORT("3mm scalar", c0, c1, (long)NI*NJ*NK + (long)NJ*NL*NM + (long)NI*NL*NJ);
  c0 = cyc();
  mm3_kernel1_ct(NDRANGE2(CEILDIV(NJ, LX), CEILDIV(NI, LY), LX, LY), A, B, E, NI, NJ, NK);
  mm3_kernel2_ct(NDRANGE2(CEILDIV(NL, LX), CEILDIV(NJ, LY), LX, LY), C, D, F, NJ, NL, NM);
  mm3_kernel3_ct(NDRANGE2(CEILDIV(NL, LX), CEILDIV(NI, LY), LX, LY), E, F, G, NI, NL, NJ);
  c1 = cyc(); REPORT("3mm hwacha-cc", c0, c1, (long)NI*NJ*NK + (long)NJ*NL*NM + (long)NI*NL*NJ);
  int bad = check_f("3mm", G, ref, NI*NL, 1e-4f);
  printf("3mm %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
