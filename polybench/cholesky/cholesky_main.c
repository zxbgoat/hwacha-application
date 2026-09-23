// PolyBench cholesky (4.2.1): in-place lower Cholesky factor of a symmetric positive-definite matrix
// (built as A*A^T of a unit lower matrix, as init_array does). Kernels: cholesky.cl, the
// right-looking form with k on the host (three launches per k); the reference is kernel_cholesky
// (row-oriented). Both apply the same subtractions in the same k order, so the factor is exact.
#include "common.h"
#define N 48
#define LS 64
void cholesky_kernel1_ct(long n, float *A, int k, int n_);
void cholesky_kernel2_ct(long n, float *A, int k, int n_);
void cholesky_kernel3_ct(long n, float *A, int k, int n_);
static float A[N*N], ref[N*N], B[N*N];
static void ref_kernel(void) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < i; j++) { for (int k = 0; k < j; k++) ref[i*N + j] -= ref[i*N + k] * ref[j*N + k]; ref[i*N + j] /= ref[j*N + j]; }
    for (int k = 0; k < i; k++) ref[i*N + i] -= ref[i*N + k] * ref[i*N + k];
    ref[i*N + i] = sqrtf(ref[i*N + i]);
  }
}
int main(void) {
  for (int i = 0; i < N; i++) { for (int j = 0; j <= i; j++) A[i*N + j] = (float)(-j % N) / N + 1; for (int j = i+1; j < N; j++) A[i*N + j] = 0; A[i*N + i] = 1; }
  for (int r = 0; r < N; r++) for (int s = 0; s < N; s++) B[r*N + s] = 0;
  for (int t = 0; t < N; t++) for (int r = 0; r < N; r++) for (int s = 0; s < N; s++) B[r*N + s] += A[r*N + t] * A[s*N + t];
  for (int i = 0; i < N*N; i++) A[i] = ref[i] = B[i];
  unsigned long c0 = cyc(); ref_kernel(); unsigned long c1 = cyc();
  REPORT("cholesky scalar", c0, c1, (long)N*N*N/6);
  c0 = cyc();
  for (int k = 0; k < N; k++) {
    cholesky_kernel1_ct(NDRANGE1(CEILDIV(N - (k+1), LS)*LS, LS), A, k, N);
    cholesky_kernel2_ct(NDRANGE1(1, 1), A, k, N);
    cholesky_kernel3_ct(NDRANGE2(CEILDIV(N - (k+1), LS), N - (k+1), LS, 1), A, k, N);
  }
  c1 = cyc(); REPORT("cholesky hwacha-cc", c0, c1, (long)N*N*N/6);
  int bad = 0; for (int i = 0; i < N; i++) for (int j = 0; j <= i; j++) { float d = fabsf(A[i*N + j] - ref[i*N + j]); if (d > 1e-4f * (fabsf(ref[i*N + j]) + 1)) bad++; }
  printf("cholesky: %d lower-triangle mismatches / %ld\n", bad, (long)N*(N+1)/2);
  printf("cholesky %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
