// PolyBench lu: in-place LU without pivoting. Kernel: the unmodified PolyBench/GPU lu.cl, whose host
// splits the classic right-looking form into two kernels per k (row scaling then the trailing update),
// with k iterated on the host. The reference is the same right-looking algorithm (lu.c of the GPU
// suite); PolyBenchC-4.2.1 writes it in the left-looking form, which factors the same matrix.
#include "common.h"
#define N 64
#define LS 64
void lu_kernel1_ct(long n, float *A, int k, int n_);
void lu_kernel2_ct(long n, float *A, int k, int n_);
static float A[N*N], ref[N*N];
static void ref_kernel(void) {
  for (int k = 0; k < N; k++) {
    for (int j = k + 1; j < N; j++) ref[k*N + j] /= ref[k*N + k];
    for (int i = k + 1; i < N; i++) for (int j = k + 1; j < N; j++) ref[i*N + j] -= ref[i*N + k] * ref[k*N + j];
  }
}
int main(void) {
  for (int i = 0; i < N; i++) { for (int j = 0; j <= i; j++) A[i*N + j] = (float)(-j % N) / N + 1; for (int j = i + 1; j < N; j++) A[i*N + j] = 0; A[i*N + i] = 1; }
  for (int i = 0; i < N*N; i++) ref[i] = A[i];
  unsigned long c0 = cyc(); ref_kernel(); unsigned long c1 = cyc();
  REPORT("lu scalar", c0, c1, (long)N*N);
  c0 = cyc();
  for (int k = 0; k < N; k++) {   // as the original host: one small launch per k
    lu_kernel1_ct(NDRANGE1(CEILDIV(N - (k+1), LS)*LS, LS), A, k, N);
    lu_kernel2_ct(NDRANGE2(CEILDIV(N - (k+1), LS), CEILDIV(N - (k+1), LS), LS, 1), A, k, N);
  }
  c1 = cyc(); REPORT("lu hwacha-cc", c0, c1, (long)N*N);
  int bad = check_f("lu", A, ref, N*N, 1e-4f);
  printf("lu %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
