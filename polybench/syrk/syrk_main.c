// PolyBench syrk: C := alpha*A*A^T + beta*C (lower triangle). Kernel: the unmodified PolyBench/GPU
// syrk.cl, which computes the full square (the reference only fills j <= i, so the comparison is on
// the lower triangle). alpha / beta and the inputs follow PolyBenchC-4.2.1's init_array.
#include "common.h"
#define N 48
#define LX 8
#define LY 8
void syrk_kernel_ct(long n, float *a, float *c, float alpha, float beta, int ni, int nj);
static float A[N*N], C[N*N], ref[N*N];
static const float alpha = 1.5f, beta = 1.2f;
static void ref_kernel(void) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j <= i; j++) ref[i*N + j] *= beta;
    for (int k = 0; k < N; k++) for (int j = 0; j <= i; j++) ref[i*N + j] += alpha * A[i*N + k] * A[j*N + k];
  }
}
int main(void) {
  for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) { A[i*N + j] = (float)((i*j+1) % N) / N; C[i*N + j] = ref[i*N + j] = (float)((i*j+2) % N) / N; }
  unsigned long c0 = cyc(); ref_kernel(); unsigned long c1 = cyc();
  REPORT("syrk scalar", c0, c1, (long)N*N);
  c0 = cyc(); syrk_kernel_ct(NDRANGE2(CEILDIV(N, LX), CEILDIV(N, LY), LX, LY), A, C, alpha, beta, N, N); c1 = cyc();
  REPORT("syrk hwacha-cc", c0, c1, (long)N*N);
  int bad = 0; for (int i = 0; i < N; i++) for (int j = 0; j <= i; j++) { float d = fabsf(C[i*N + j] - ref[i*N + j]); if (d > 1e-4f * (fabsf(ref[i*N + j]) + 1)) bad++; }
  printf("syrk: %d lower-triangle mismatches / %ld\n", bad, (long)N*(N+1)/2);
  printf("syrk %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
