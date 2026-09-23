// PolyBench doitgen (4.2.1): A[r][q][:] := A[r][q][:] * C4 for every (r, q). Kernel: doitgen.cl (one
// work-item per (r, q), sum in a global scratch array as in kernel_doitgen).
#include "common.h"
#define NR 16
#define NQ 16
#define NP 16
#define LX 8
#define LY 8
void doitgen_kernel_ct(long n, float *A, float *C4, float *sum, int nr, int nq, int np);
static float A[NR*NQ*NP], C4[NP*NP], sum[NR*NQ*NP], ref[NR*NQ*NP], rsum[NP];
static void ref_kernel(void) {
  for (int r = 0; r < NR; r++) for (int q = 0; q < NQ; q++) {
    for (int p = 0; p < NP; p++) { rsum[p] = 0; for (int s = 0; s < NP; s++) rsum[p] += ref[(r*NQ + q)*NP + s] * C4[s*NP + p]; }
    for (int p = 0; p < NP; p++) ref[(r*NQ + q)*NP + p] = rsum[p];
  }
}
int main(void) {
  for (int i = 0; i < NR; i++) for (int j = 0; j < NQ; j++) for (int k = 0; k < NP; k++) A[(i*NQ + j)*NP + k] = ref[(i*NQ + j)*NP + k] = (float)((i*j + k) % NP) / NP;
  for (int i = 0; i < NP; i++) for (int j = 0; j < NP; j++) C4[i*NP + j] = (float)(i*j % NP) / NP;
  unsigned long c0 = cyc(); ref_kernel(); unsigned long c1 = cyc();
  REPORT("doitgen scalar", c0, c1, (long)NR*NQ*NP*NP);
  c0 = cyc(); doitgen_kernel_ct(NDRANGE2(CEILDIV(NQ, LX), CEILDIV(NR, LY), LX, LY), A, C4, sum, NR, NQ, NP); c1 = cyc();
  REPORT("doitgen hwacha-cc", c0, c1, (long)NR*NQ*NP*NP);
  int bad = check_f("doitgen", A, ref, NR*NQ*NP, 1e-4f);
  printf("doitgen %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
