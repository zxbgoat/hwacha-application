// PolyBench adi: TSTEPS sweeps of the alternating-direction implicit solver (row forward/back
// substitution, then column). Kernels: the unmodified PolyBench/GPU adi.cl, whose N is a compile-time
// constant (#ifndef N, the Makefile passes -DN=64), launched in the original host's order: kernel1/2/3
// once per step (one work-item per row), kernel4 for i1 = 1..N-1, kernel5 once, kernel6 for i1 =
// 0..N-3 (one work-item per column). Reference: adi() of that host.
#include "common.h"
#define N 64
#define TSTEPS 2
#define LS 64
void adi_kernel1_ct(long n, float *A, float *B, float *X);
void adi_kernel2_ct(long n, float *A, float *B, float *X);
void adi_kernel3_ct(long n, float *A, float *B, float *X);
void adi_kernel4_ct(long n, float *A, float *B, float *X, int i1);
void adi_kernel5_ct(long n, float *A, float *B, float *X);
void adi_kernel6_ct(long n, float *A, float *B, float *X, int i1);
static float A[N*N], B[N*N], X[N*N], rB[N*N], rX[N*N];
#define I(i, j) ((i)*N + (j))
static void ref_kernel(void) {
  for (int t = 0; t < TSTEPS; t++) {
    for (int i1 = 0; i1 < N; i1++) for (int i2 = 1; i2 < N; i2++) {
      rX[I(i1,i2)] = rX[I(i1,i2)] - rX[I(i1,i2-1)] * A[I(i1,i2)] / rB[I(i1,i2-1)];
      rB[I(i1,i2)] = rB[I(i1,i2)] - A[I(i1,i2)] * A[I(i1,i2)] / rB[I(i1,i2-1)]; }
    for (int i1 = 0; i1 < N; i1++) rX[I(i1,N-1)] = rX[I(i1,N-1)] / rB[I(i1,N-1)];
    for (int i1 = 0; i1 < N; i1++) for (int i2 = 0; i2 < N-2; i2++)
      rX[I(i1,N-i2-2)] = (rX[I(i1,N-2-i2)] - rX[I(i1,N-2-i2-1)] * A[I(i1,N-i2-3)]) / rB[I(i1,N-3-i2)];
    for (int i1 = 1; i1 < N; i1++) for (int i2 = 0; i2 < N; i2++) {
      rX[I(i1,i2)] = rX[I(i1,i2)] - rX[I(i1-1,i2)] * A[I(i1,i2)] / rB[I(i1-1,i2)];
      rB[I(i1,i2)] = rB[I(i1,i2)] - A[I(i1,i2)] * A[I(i1,i2)] / rB[I(i1-1,i2)]; }
    for (int i2 = 0; i2 < N; i2++) rX[I(N-1,i2)] = rX[I(N-1,i2)] / rB[I(N-1,i2)];
    for (int i1 = 0; i1 < N-2; i1++) for (int i2 = 0; i2 < N; i2++)
      rX[I(N-2-i1,i2)] = (rX[I(N-2-i1,i2)] - rX[I(N-i1-3,i2)] * A[I(N-3-i1,i2)]) / rB[I(N-2-i1,i2)];
  }
}
int main(void) {
  for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
    X[I(i,j)] = rX[I(i,j)] = ((float)(i*(j+1) + 1)) / N; A[I(i,j)] = ((float)((i-1)*(j+4) + 2)) / N; B[I(i,j)] = rB[I(i,j)] = ((float)((i+3)*(j+7) + 3)) / N; }
  unsigned long c0 = cyc(); ref_kernel(); unsigned long c1 = cyc();
  REPORT("adi scalar", c0, c1, (long)N*N*TSTEPS);
  c0 = cyc();
  for (int t = 0; t < TSTEPS; t++) {
    adi_kernel1_ct(NDRANGE1(N, LS), A, B, X);
    adi_kernel2_ct(NDRANGE1(N, LS), A, B, X);
    adi_kernel3_ct(NDRANGE1(N, LS), A, B, X);
    for (int i1 = 1; i1 < N; i1++) adi_kernel4_ct(NDRANGE1(N, LS), A, B, X, i1);
    adi_kernel5_ct(NDRANGE1(N, LS), A, B, X);
    for (int i1 = 0; i1 < N - 2; i1++) adi_kernel6_ct(NDRANGE1(N, LS), A, B, X, i1);
  }
  c1 = cyc(); REPORT("adi hwacha-cc", c0, c1, (long)N*N*TSTEPS);
  int bad = check_f("adi X", X, rX, N*N, 1e-3f);
  bad += check_f("adi B", B, rB, N*N, 1e-3f);
  printf("adi %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
