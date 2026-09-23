// PolyBench gesummv: y := alpha*A*x + beta*B*x. Kernel: the unmodified PolyBench/GPU gesummv.cl
// (tmp[i] and y[i] are accumulated by the kernel, so both are seeded here as the original host does);
// reference: the CPU version in that same file. alpha / beta follow PolyBenchC-4.2.1's init_array.
#include "common.h"
#define N 64
#define LS 64
void gesummv_kernel_ct(long n, float *a, float *b, float *x, float *y, float *tmp, float alpha, float beta, int n_);
static float A[N*N], B[N*N], x[N], y[N], tmp[N], ref[N];
static const float alpha = 1.5f, beta = 1.2f;
static void ref_kernel(void) {
  for (int i = 0; i < N; i++) {
    tmp[i] = 0; y[i] = 0;
    for (int j = 0; j < N; j++) { tmp[i] += A[i*N + j] * x[j]; y[i] += B[i*N + j] * x[j]; }
    ref[i] = alpha * tmp[i] + beta * y[i];
  }
}
int main(void) {
  for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) { A[i*N + j] = (float)((i*j+1) % N) / N; B[i*N + j] = (float)((i*j+2) % N) / N; }
  for (int i = 0; i < N; i++) x[i] = (float)((i % N) + 1) / N;
  for (int i = 0; i < N; i++) { y[i] = 0; tmp[i] = 0; }
  unsigned long c0 = cyc(); ref_kernel(); unsigned long c1 = cyc();
  REPORT("gesummv scalar", c0, c1, (long)N*N);
  for (int i = 0; i < N; i++) { y[i] = 0; tmp[i] = 0; }
  c0 = cyc(); gesummv_kernel_ct(NDRANGE1(N, LS), A, B, x, y, tmp, alpha, beta, N); c1 = cyc();
  REPORT("gesummv hwacha-cc", c0, c1, (long)N*N);
  int bad = check_f("gesummv", y, ref, N, 1e-4f);
  printf("gesummv %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
