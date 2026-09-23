// PolyBench gemver: A += u1 v1^T + u2 v2^T; x += beta A^T y + z; w += alpha A x. Three kernels, the
// unmodified PolyBench/GPU gemver.cl. Note that kernel1 adds to A in place (the GPU host passes the
// array it also uses as the reference input), kernel2 adds to x and kernel3 to w, so every buffer is
// seeded exactly as the original host seeds it and the reference adds to the same values.
#include "common.h"
#define N 64
#define LS 64
void gemver_kernel1_ct(long n, float *A, float *V1, float *V2, float *U1, float *U2, int n_);
void gemver_kernel2_ct(long n, float *A, float *X, float *Y, float *Z, float beta, int n_);
void gemver_kernel3_ct(long n, float *A, float *X, float *w, float alpha, int n_);
static float A[N*N], B[N*N], x[N], y[N], z[N], w[N], u1[N], u2[N], v1[N], v2[N];
static float refx[N], refw[N];
static const float alpha = 1.5f, beta = 1.2f;   // PolyBenchC-4.2.1 init_array
static void ref_kernel(void) {
  for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) A[i*N + j] = A[i*N + j] + u1[i]*v1[j] + u2[i]*v2[j];
  for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) refx[i] = refx[i] + beta * A[j*N + i] * y[j];
  for (int i = 0; i < N; i++) refx[i] = refx[i] + z[i];
  for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) refw[i] = refw[i] + alpha * A[i*N + j] * refx[j];
}
int main(void) {
  for (int i = 0; i < N; i++) {
    u1[i] = i; u2[i] = (i+1)/N/2.0f; v1[i] = (i+1)/N/4.0f; v2[i] = (i+1)/N/6.0f;
    y[i] = (i+1)/N/8.0f; z[i] = (i+1)/N/9.0f; refx[i] = x[i] = 0.0f; refw[i] = w[i] = 0.0f;
    for (int j = 0; j < N; j++) A[i*N + j] = B[i*N + j] = (float)(i*j) / N;
  }
  unsigned long c0 = cyc(); ref_kernel(); unsigned long c1 = cyc();
  REPORT("gemver scalar", c0, c1, (long)N*N);
  for (int i = 0; i < N*N; i++) A[i] = B[i];
  for (int i = 0; i < N; i++) { x[i] = 0; w[i] = 0; }
  // the reference consumed A in place; restore the post-kernel1 value the kernel will produce
  for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) A[i*N + j] = A[i*N + j] + u1[i]*v1[j] + u2[i]*v2[j];
  c0 = cyc();
  gemver_kernel1_ct(NDRANGE2(CEILDIV(N, LS), CEILDIV(N, LS), LS, 1), A, v1, v2, u1, u2, N);
  gemver_kernel2_ct(NDRANGE1(N, LS), A, x, y, z, beta, N);
  gemver_kernel3_ct(NDRANGE1(N, LS), A, x, w, alpha, N);
  c1 = cyc(); REPORT("gemver hwacha-cc", c0, c1, (long)N*N);
  int bad = check_f("gemver x", x, refx, N, 1e-4f);
  bad += check_f("gemver w", w, refw, N, 1e-4f);
  printf("gemver %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
