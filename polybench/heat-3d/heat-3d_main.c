// PolyBench heat-3d (4.2.1): TSTEPS of the 7-point 3-D heat stencil, two half-steps per step
// (A -> B, B -> A). Kernel: heat-3d.cl, one work-item per interior (i, j) with the k loop inside,
// launched twice per step with the arrays swapped; reference kernel_heat_3d.
#include "common.h"
#define N 16
#define TSTEPS 4
#define LX 8
#define LY 8
void heat_3d_kernel_ct(long n, float *A, float *B, int n_);
static float A[N*N*N], B[N*N*N], rA[N*N*N], rB[N*N*N];
#define I(i, j, k) (((i)*N + (j))*N + (k))
static void step(float *X, float *Y) {
  for (int i = 1; i < N-1; i++) for (int j = 1; j < N-1; j++) for (int k = 1; k < N-1; k++)
    Y[I(i,j,k)] = 0.125f * (X[I(i+1,j,k)] - 2.0f * X[I(i,j,k)] + X[I(i-1,j,k)])
                + 0.125f * (X[I(i,j+1,k)] - 2.0f * X[I(i,j,k)] + X[I(i,j-1,k)])
                + 0.125f * (X[I(i,j,k+1)] - 2.0f * X[I(i,j,k)] + X[I(i,j,k-1)])
                + X[I(i,j,k)];
}
static void ref_kernel(void) { for (int t = 1; t <= TSTEPS; t++) { step(rA, rB); step(rB, rA); } }
int main(void) {
  for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) for (int k = 0; k < N; k++) A[I(i,j,k)] = B[I(i,j,k)] = rA[I(i,j,k)] = rB[I(i,j,k)] = (float)(i + j + (N-k)) * 10 / N;
  unsigned long c0 = cyc(); ref_kernel(); unsigned long c1 = cyc();
  REPORT("heat-3d scalar", c0, c1, (long)N*N*N*TSTEPS*2);
  c0 = cyc();
  for (int t = 1; t <= TSTEPS; t++) {
    heat_3d_kernel_ct(NDRANGE2(CEILDIV(N-2, LX), CEILDIV(N-2, LY), LX, LY), A, B, N);
    heat_3d_kernel_ct(NDRANGE2(CEILDIV(N-2, LX), CEILDIV(N-2, LY), LX, LY), B, A, N);
  }
  c1 = cyc(); REPORT("heat-3d hwacha-cc", c0, c1, (long)N*N*N*TSTEPS*2);
  int bad = check_f("heat-3d A", A, rA, N*N*N, 1e-4f);
  bad += check_f("heat-3d B", B, rB, N*N*N, 1e-4f);
  printf("heat-3d %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
