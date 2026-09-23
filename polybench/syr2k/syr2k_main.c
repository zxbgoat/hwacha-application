// PolyBench syr2k: C := alpha*A*B^T + alpha*B*A^T + beta*C (lower triangle). Kernel: the unmodified
// PolyBench/GPU syr2k.cl (computes the full square; the comparison is on the lower triangle).
#include "common.h"
#define N 48
#define M 48
#define LX 8
#define LY 8
void syr2k_kernel_ct(long n, float *a, float *b, float *c, float alpha, float beta, int ni, int nj);
static float A[N*M], B[N*M], C[N*N], ref[N*N];
static const float alpha = 1.5f, beta = 1.2f;
static void ref_kernel(void) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j <= i; j++) ref[i*N + j] *= beta;
    for (int k = 0; k < M; k++) for (int j = 0; j <= i; j++) ref[i*N + j] += alpha * A[i*M + k] * B[j*M + k] + alpha * B[i*M + k] * A[j*M + k];
  }
}
int main(void) {
  for (int i = 0; i < N; i++) for (int j = 0; j < M; j++) { A[i*M + j] = (float)((i*j+1) % M) / M; B[i*M + j] = (float)((i*j+3) % M) / M; }
  for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) C[i*N + j] = ref[i*N + j] = (float)((i*j+2) % N) / N;
  unsigned long c0 = cyc(); ref_kernel(); unsigned long c1 = cyc();
  REPORT("syr2k scalar", c0, c1, (long)N*M);
  c0 = cyc(); syr2k_kernel_ct(NDRANGE2(CEILDIV(N, LX), CEILDIV(N, LY), LX, LY), A, B, C, alpha, beta, M, N); c1 = cyc();
  REPORT("syr2k hwacha-cc", c0, c1, (long)N*M);
  int bad = 0; for (int i = 0; i < N; i++) for (int j = 0; j <= i; j++) { float d = fabsf(C[i*N + j] - ref[i*N + j]); if (d > 1e-4f * (fabsf(ref[i*N + j]) + 1)) bad++; }
  printf("syr2k: %d lower-triangle mismatches / %ld\n", bad, (long)N*(N+1)/2);
  printf("syr2k %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
