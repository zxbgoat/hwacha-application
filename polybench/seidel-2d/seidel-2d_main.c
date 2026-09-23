// PolyBench seidel-2d (4.2.1): TSTEPS Gauss-Seidel sweeps of the 9-point average, in place. Kernel:
// seidel-2d.cl, one launch per anti-diagonal i + j = d per sweep (the entries of a diagonal are
// independent given the previous ones), one work-item per entry; reference kernel_seidel_2d.
#include "common.h"
#define N 32
#define TSTEPS 2
#define LS 64
void seidel_2d_kernel_ct(long n, float *A, int d, int n_);
static float A[N*N], ref[N*N];
static void ref_kernel(void) {
  for (int t = 0; t <= TSTEPS - 1; t++) for (int i = 1; i <= N - 2; i++) for (int j = 1; j <= N - 2; j++)
    ref[i*N + j] = (ref[(i-1)*N + (j-1)] + ref[(i-1)*N + j] + ref[(i-1)*N + (j+1)] + ref[i*N + (j-1)] + ref[i*N + j] + ref[i*N + (j+1)] + ref[(i+1)*N + (j-1)] + ref[(i+1)*N + j] + ref[(i+1)*N + (j+1)]) / 9.0f;
}
int main(void) {
  for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) A[i*N + j] = ref[i*N + j] = ((float)(i*(j+2) + 2)) / N;
  unsigned long c0 = cyc(); ref_kernel(); unsigned long c1 = cyc();
  REPORT("seidel-2d scalar", c0, c1, (long)N*N*TSTEPS);
  c0 = cyc();
  for (int t = 0; t < TSTEPS; t++) for (int d = 2; d <= 2*(N-2); d++) seidel_2d_kernel_ct(NDRANGE1(CEILDIV(N-2, LS)*LS, LS), A, d, N);
  c1 = cyc(); REPORT("seidel-2d hwacha-cc", c0, c1, (long)N*N*TSTEPS);
  int bad = check_f("seidel-2d", A, ref, N*N, 1e-4f);
  printf("seidel-2d %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
