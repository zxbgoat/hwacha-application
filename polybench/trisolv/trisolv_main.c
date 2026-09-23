// PolyBench trisolv (4.2.1): forward substitution L x = b. Kernel: trisolv.cl, column-oriented with
// one launch per row i (the work-item of row i finishes x[i], the others subtract column i from
// the rows below); every row subtracts in the same column order as kernel_trisolv, so exact.
#include "common.h"
#define N 64
#define LS 64
void trisolv_kernel_ct(long n, float *L, float *x, float *b, int i, int n_);
static float L[N*N], x[N], b[N], bw[N], ref[N];
static void ref_kernel(void) {
  for (int i = 0; i < N; i++) { ref[i] = b[i]; for (int j = 0; j < i; j++) ref[i] -= L[i*N + j] * ref[j]; ref[i] = ref[i] / L[i*N + i]; }
}
int main(void) {
  for (int i = 0; i < N; i++) { x[i] = -999; b[i] = bw[i] = i; for (int j = 0; j <= i; j++) L[i*N + j] = (float)(i+N-j+1)*2/N; }
  unsigned long c0 = cyc(); ref_kernel(); unsigned long c1 = cyc();
  REPORT("trisolv scalar", c0, c1, (long)N*N);
  c0 = cyc();
  for (int i = 0; i < N; i++) trisolv_kernel_ct(NDRANGE1(CEILDIV(N - i, LS)*LS, LS), L, x, bw, i, N);
  c1 = cyc(); REPORT("trisolv hwacha-cc", c0, c1, (long)N*N);
  int bad = check_f("trisolv", x, ref, N, 1e-4f);
  printf("trisolv %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
