// PolyBench durbin (4.2.1): Levinson-Durbin recursion for a Toeplitz system. Kernels: durbin.cl,
// three launches per step k (one work-item advances alpha / beta / sum, k work-items form z, k+1
// copy it back); reference kernel_durbin. Same operations in the same order: exact.
#include "common.h"
#define N 64
#define LS 64
void durbin_kernel1_ct(long n, float *r, float *y, float *st, int k, int n_);
void durbin_kernel2_ct(long n, float *y, float *z, float *st, int k);
void durbin_kernel3_ct(long n, float *y, float *z, float *st, int k);
static float r[N], y[N], z[N], st[2], ref[N], rz[N];
static void ref_kernel(void) {
  ref[0] = -r[0]; float beta = 1.0f, alpha = -r[0];
  for (int k = 1; k < N; k++) {
    beta = (1 - alpha*alpha) * beta;
    float sum = 0; for (int i = 0; i < k; i++) sum += r[k-i-1] * ref[i];
    alpha = -(r[k] + sum) / beta;
    for (int i = 0; i < k; i++) rz[i] = ref[i] + alpha * ref[k-i-1];
    for (int i = 0; i < k; i++) ref[i] = rz[i];
    ref[k] = alpha;
  }
}
int main(void) {
  for (int i = 0; i < N; i++) r[i] = (N + 1 - i);
  unsigned long c0 = cyc(); ref_kernel(); unsigned long c1 = cyc();
  REPORT("durbin scalar", c0, c1, (long)N*N);
  y[0] = -r[0]; st[0] = -r[0]; st[1] = 1.0f;
  c0 = cyc();
  for (int k = 1; k < N; k++) {
    durbin_kernel1_ct(NDRANGE1(1, 1), r, y, st, k, N);
    durbin_kernel2_ct(NDRANGE1(CEILDIV(k, LS)*LS, LS), y, z, st, k);
    durbin_kernel3_ct(NDRANGE1(CEILDIV(k + 1, LS)*LS, LS), y, z, st, k);
  }
  c1 = cyc(); REPORT("durbin hwacha-cc", c0, c1, (long)N*N);
  int bad = check_f("durbin", y, ref, N, 1e-4f);
  printf("durbin %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
