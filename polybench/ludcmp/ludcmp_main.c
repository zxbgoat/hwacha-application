// PolyBench ludcmp (4.2.1): LU factorisation (unit lower L) then forward and back substitution of
// A x = b. Kernels: ludcmp.cl -- right-looking LU with k on the host (exact), column-oriented
// forward substitution (exact) and back substitution (subtracts in the opposite order to
// kernel_ludcmp: rounding-level differences, hence the tolerance).
#include "common.h"
#define N 48
#define LS 64
void ludcmp_kernel1_ct(long n, float *A, int k, int n_);
void ludcmp_kernel2_ct(long n, float *A, int k, int n_);
void ludcmp_kernel3_ct(long n, float *A, float *w, float *y, int i, int n_);
void ludcmp_kernel4_ct(long n, float *A, float *w, float *x, int i, int n_);
static float A[N*N], b[N], x[N], y[N], w[N], refA[N*N], refx[N], refy[N], B[N*N];
static void ref_kernel(void) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < i; j++) { float ww = refA[i*N + j]; for (int k = 0; k < j; k++) ww -= refA[i*N + k] * refA[k*N + j]; refA[i*N + j] = ww / refA[j*N + j]; }
    for (int j = i; j < N; j++) { float ww = refA[i*N + j]; for (int k = 0; k < i; k++) ww -= refA[i*N + k] * refA[k*N + j]; refA[i*N + j] = ww; }
  }
  for (int i = 0; i < N; i++) { float ww = b[i]; for (int j = 0; j < i; j++) ww -= refA[i*N + j] * refy[j]; refy[i] = ww; }
  for (int i = N-1; i >= 0; i--) { float ww = refy[i]; for (int j = i+1; j < N; j++) ww -= refA[i*N + j] * refx[j]; refx[i] = ww / refA[i*N + i]; }
}
int main(void) {
  for (int i = 0; i < N; i++) { x[i] = refx[i] = 0; y[i] = refy[i] = 0; b[i] = (i+1)/(float)N/2.0f + 4; }
  for (int i = 0; i < N; i++) { for (int j = 0; j <= i; j++) A[i*N + j] = (float)(-j % N) / N + 1; for (int j = i+1; j < N; j++) A[i*N + j] = 0; A[i*N + i] = 1; }
  for (int r = 0; r < N; r++) for (int s = 0; s < N; s++) B[r*N + s] = 0;
  for (int t = 0; t < N; t++) for (int r = 0; r < N; r++) for (int s = 0; s < N; s++) B[r*N + s] += A[r*N + t] * A[s*N + t];
  for (int i = 0; i < N*N; i++) A[i] = refA[i] = B[i];
  unsigned long c0 = cyc(); ref_kernel(); unsigned long c1 = cyc();
  REPORT("ludcmp scalar", c0, c1, (long)N*N*N/3);
  c0 = cyc();
  for (int k = 0; k < N; k++) {
    ludcmp_kernel1_ct(NDRANGE1(CEILDIV(N - (k+1), LS)*LS, LS), A, k, N);
    ludcmp_kernel2_ct(NDRANGE2(CEILDIV(N - (k+1), LS), N - (k+1), LS, 1), A, k, N);
  }
  for (int i = 0; i < N; i++) w[i] = b[i];
  for (int i = 0; i < N; i++) ludcmp_kernel3_ct(NDRANGE1(CEILDIV(N - i, LS)*LS, LS), A, w, y, i, N);
  for (int i = 0; i < N; i++) w[i] = y[i];
  for (int i = N-1; i >= 0; i--) ludcmp_kernel4_ct(NDRANGE1(CEILDIV(i + 1, LS)*LS, LS), A, w, x, i, N);
  c1 = cyc(); REPORT("ludcmp hwacha-cc", c0, c1, (long)N*N*N/3);
  int bad = check_f("ludcmp A", A, refA, N*N, 1e-4f);
  bad += check_f("ludcmp y", y, refy, N, 1e-4f);
  bad += check_f("ludcmp x", x, refx, N, 1e-3f);
  printf("ludcmp %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
