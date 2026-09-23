// PolyBench jacobi-1d: TSTEPS of B[i] = (A[i-1]+A[i]+A[i+1])/3 (the kernel writes 0.33333*, as the
// PolyBench/GPU version does) then A = B, with the boundary elements untouched. Kernels: the
// unmodified PolyBench/GPU jacobi1D.cl, one pair of launches per step.
#include "common.h"
#define N 128
#define TSTEPS 8
#define LS 64
void runJacobi1D_kernel1_ct(long n, float *A, float *B, int n_);
void runJacobi1D_kernel2_ct(long n, float *A, float *B, int n_);
static float A[N], B[N], ref[N];
static void ref_kernel(void) {
  for (int t = 0; t < TSTEPS; t++) {
    for (int i = 1; i < N - 1; i++) B[i] = 0.33333f * (ref[i-1] + ref[i] + ref[i+1]);
    for (int i = 1; i < N - 1; i++) ref[i] = B[i];
  }
}
int main(void) {
  for (int i = 0; i < N; i++) { A[i] = ref[i] = (float)(4*i + 10) / N; B[i] = (float)(7*i + 11) / N; }
  unsigned long c0 = cyc(); ref_kernel(); unsigned long c1 = cyc();
  REPORT("jacobi-1d scalar", c0, c1, (long)N*TSTEPS);
  c0 = cyc();
  for (int t = 0; t < TSTEPS; t++) {
    runJacobi1D_kernel1_ct(NDRANGE1(CEILDIV(N, LS)*LS, LS), A, B, N);
    runJacobi1D_kernel2_ct(NDRANGE1(CEILDIV(N, LS)*LS, LS), A, B, N);
  }
  c1 = cyc(); REPORT("jacobi-1d hwacha-cc", c0, c1, (long)N*TSTEPS);

  int bad = check_f("jacobi-1d A", A, ref, N, 1e-4f);   // B keeps its boundary elements: only A is the live-out
  printf("jacobi-1d %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
