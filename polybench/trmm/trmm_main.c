// PolyBench trmm (4.2.1): B := alpha*A^T*B, A unit lower triangular. Kernel: trmm.cl (one
// work-item per B element, reading the untouched input B as kernel_trmm's loop order does).
#include "common.h"
#define M 32
#define N 40
#define LX 8
#define LY 8
void trmm_kernel_ct(long n, float *A, float *Bin, float *Bout, float alpha, int m, int n_);
static float A[M*M], B[M*N], Bout[M*N], ref[M*N];
static const float alpha = 1.5f;
static void ref_kernel(void) {
  for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) { for (int k = i+1; k < M; k++) ref[i*N + j] += A[k*M + i] * ref[k*N + j]; ref[i*N + j] = alpha * ref[i*N + j]; }
}
int main(void) {
  for (int i = 0; i < M; i++) { for (int j = 0; j < i; j++) A[i*M + j] = (float)((i+j) % M) / M; A[i*M + i] = 1.0f; for (int j = 0; j < N; j++) B[i*N + j] = ref[i*N + j] = (float)((N+(i-j)) % N) / N; }
  unsigned long c0 = cyc(); ref_kernel(); unsigned long c1 = cyc();
  REPORT("trmm scalar", c0, c1, (long)M*M*N);
  c0 = cyc(); trmm_kernel_ct(NDRANGE2(CEILDIV(N, LX), CEILDIV(M, LY), LX, LY), A, B, Bout, alpha, M, N); c1 = cyc();
  REPORT("trmm hwacha-cc", c0, c1, (long)M*M*N);
  int bad = check_f("trmm", Bout, ref, M*N, 1e-4f);
  printf("trmm %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
