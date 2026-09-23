// PolyBench mvt: x1 := A y1 (rows) and x2 := A^T y2 (columns), two kernels. The kernel is the
// unmodified PolyBench/GPU mvt.cl (each work-item accumulates into its own output element).
#include "common.h"
#define N 64
#define LS 64
void mvt_kernel1_ct(long n, float *a, float *x1, float *yy1, int n_);
void mvt_kernel2_ct(long n, float *a, float *x2, float *y2, int n_);
static float A[N*N], x1[N], x2[N], yy1[N], y2[N], ref1[N], ref2[N];
static void ref_kernel(void) {
  for (int i = 0; i < N; i++) { for (int j = 0; j < N; j++) { ref1[i] += A[i*N + j] * yy1[j]; ref2[i] += A[j*N + i] * y2[j]; } }
}
int main(void) {
  for (int i = 0; i < N; i++) { for (int j = 0; j < N; j++) A[i*N + j] = (float)((i+j+1) % N) / N; yy1[i] = (float)(i % N) / N; y2[i] = (float)((i+1) % N) / N; x1[i] = x2[i] = ref1[i] = ref2[i] = 0; }
  unsigned long c0 = cyc(); ref_kernel(); unsigned long c1 = cyc();
  REPORT("mvt scalar", c0, c1, (long)N*N);
  c0 = cyc(); mvt_kernel1_ct(NDRANGE1(N, LS), A, x1, yy1, N); mvt_kernel2_ct(NDRANGE1(N, LS), A, x2, y2, N); c1 = cyc();
  REPORT("mvt hwacha-cc", c0, c1, (long)N*N);
  int bad = check_f("mvt x1", x1, ref1, N, 1e-4f);
  bad += check_f("mvt x2", x2, ref2, N, 1e-4f);
  printf("mvt %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
