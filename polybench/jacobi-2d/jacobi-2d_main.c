// PolyBench jacobi-2d: TSTEPS of the 5-point average into B then A = B. Kernels: the unmodified
// PolyBench/GPU jacobi2D.cl, one pair of 2-D launches per step.
#include "common.h"
#define N 64
#define TSTEPS 4
#define LX 8
#define LY 8
void runJacobi2D_kernel1_ct(long n, float *A, float *B, int n_);
void runJacobi2D_kernel2_ct(long n, float *A, float *B, int n_);
static float A[N*N], B[N*N], ref[N*N];
static void ref_kernel(void) {
  for (int t = 0; t < TSTEPS; t++) {
    for (int i = 1; i < N - 1; i++) for (int j = 1; j < N - 1; j++)
      B[i*N + j] = 0.2f * (ref[i*N + j] + ref[i*N + (j-1)] + ref[i*N + (j+1)] + ref[(i+1)*N + j] + ref[(i-1)*N + j]);
    for (int i = 1; i < N - 1; i++) for (int j = 1; j < N - 1; j++) ref[i*N + j] = B[i*N + j];
  }
}
int main(void) {
  for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) { A[i*N + j] = ref[i*N + j] = ((float)(i*(j+2) + 10)) / N; B[i*N + j] = ((float)((i-4)*(j-1) + 11)) / N; }
  unsigned long c0 = cyc(); ref_kernel(); unsigned long c1 = cyc();
  REPORT("jacobi-2d scalar", c0, c1, (long)N*N*TSTEPS);
  c0 = cyc();
  for (int t = 0; t < TSTEPS; t++) {
    runJacobi2D_kernel1_ct(NDRANGE2(CEILDIV(N, LX), CEILDIV(N, LY), LX, LY), A, B, N);
    runJacobi2D_kernel2_ct(NDRANGE2(CEILDIV(N, LX), CEILDIV(N, LY), LX, LY), A, B, N);
  }
  c1 = cyc(); REPORT("jacobi-2d hwacha-cc", c0, c1, (long)N*N*TSTEPS);
  int bad = check_f("jacobi-2d A", A, ref, N*N, 1e-4f);
  printf("jacobi-2d %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
