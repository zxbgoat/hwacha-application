// PolyBench symm (4.2.1): C := alpha*A*B + beta*C with A symmetric, lower triangle stored. Kernel:
// symm.cl (one work-item per C element, same operation order as kernel_symm); inputs and the
// reference from PolyBenchC-4.2.1.
#include "common.h"
#define M 32
#define N 40
#define LX 8
#define LY 8
void symm_kernel_ct(long n, float *C, float *A, float *B, float alpha, float beta, int m, int n_);
static float C[M*N], A[M*M], B[M*N], ref[M*N];
static const float alpha = 1.5f, beta = 1.2f;
static void ref_kernel(void) {
  for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {
    float temp2 = 0;
    for (int k = 0; k < i; k++) { ref[k*N + j] += alpha*B[i*N + j] * A[i*M + k]; temp2 += B[k*N + j] * A[i*M + k]; }
    ref[i*N + j] = beta * ref[i*N + j] + alpha*B[i*N + j] * A[i*M + i] + alpha * temp2;
  }
}
int main(void) {
  for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) { C[i*N + j] = ref[i*N + j] = (float)((i+j) % 100) / M; B[i*N + j] = (float)((N+i-j) % 100) / M; }
  for (int i = 0; i < M; i++) { for (int j = 0; j <= i; j++) A[i*M + j] = (float)((i+j) % 100) / M; for (int j = i+1; j < M; j++) A[i*M + j] = -999; }
  unsigned long c0 = cyc(); ref_kernel(); unsigned long c1 = cyc();
  REPORT("symm scalar", c0, c1, (long)M*M*N);
  c0 = cyc(); symm_kernel_ct(NDRANGE2(CEILDIV(N, LX), CEILDIV(M, LY), LX, LY), C, A, B, alpha, beta, M, N); c1 = cyc();
  REPORT("symm hwacha-cc", c0, c1, (long)M*M*N);
  int bad = check_f("symm", C, ref, M*N, 1e-4f);
  printf("symm %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
