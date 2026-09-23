// PolyBench correlation (corr): column means and standard deviations, centre-and-reduce, then the
// M x M correlation matrix. Kernels: the unmodified PolyBench/GPU correlation.cl (mean, std, reduce,
// corr), launched in the original host's order; corr_kernel accumulates into symmat, so symmat is
// zero-seeded, and its last diagonal element is set on the host as the original host does.
// Inputs, float_n = N and eps = 0.1 follow PolyBenchC-4.2.1's init_array / kernel_correlation.
#include "common.h"
#define M 32
#define N 40
#define LS 64
#define LX 8
#define LY 8
void mean_kernel_ct(long n, float *mean, float *data, float float_n, int m, int n_);
void std_kernel_ct(long n, float *mean, float *std, float *data, float float_n, float eps, int m, int n_);
void reduce_kernel_ct(long n, float *mean, float *std, float *data, float float_n, int m, int n_);
void corr_kernel_ct(long n, float *symmat, float *data, int m, int n_);
static float data[N*M], mean[M], stddev[M], symmat[M*M], rdata[N*M], rmean[M], rstd[M], ref[M*M];
static const float float_n = (float)N, eps = 0.1f;
static void ref_kernel(void) {
  for (int j = 0; j < M; j++) { rmean[j] = 0; for (int i = 0; i < N; i++) rmean[j] += rdata[i*M + j]; rmean[j] /= float_n; }
  for (int j = 0; j < M; j++) { rstd[j] = 0; for (int i = 0; i < N; i++) rstd[j] += (rdata[i*M + j] - rmean[j]) * (rdata[i*M + j] - rmean[j]); rstd[j] /= float_n; rstd[j] = sqrtf(rstd[j]); rstd[j] = rstd[j] <= eps ? 1.0f : rstd[j]; }
  for (int i = 0; i < N; i++) for (int j = 0; j < M; j++) { rdata[i*M + j] -= rmean[j]; rdata[i*M + j] /= (sqrtf(float_n) * rstd[j]); }
  for (int j1 = 0; j1 < M - 1; j1++) { ref[j1*M + j1] = 1.0f; for (int j2 = j1 + 1; j2 < M; j2++) { ref[j1*M + j2] = 0; for (int i = 0; i < N; i++) ref[j1*M + j2] += rdata[i*M + j1] * rdata[i*M + j2]; ref[j2*M + j1] = ref[j1*M + j2]; } }
  ref[(M-1)*M + (M-1)] = 1.0f;
}
int main(void) {
  for (int i = 0; i < N; i++) for (int j = 0; j < M; j++) data[i*M + j] = rdata[i*M + j] = (float)(i*j) / M + i;
  for (int i = 0; i < M*M; i++) symmat[i] = 0;
  unsigned long c0 = cyc(); ref_kernel(); unsigned long c1 = cyc();
  REPORT("corr scalar", c0, c1, (long)M*M*N);
  c0 = cyc();
  mean_kernel_ct(NDRANGE1(CEILDIV(M, LS)*LS, LS), mean, data, float_n, M, N);
  std_kernel_ct(NDRANGE1(CEILDIV(M, LS)*LS, LS), mean, stddev, data, float_n, eps, M, N);
  reduce_kernel_ct(NDRANGE2(CEILDIV(M, LX), CEILDIV(N, LY), LX, LY), mean, stddev, data, float_n, M, N);
  corr_kernel_ct(NDRANGE1(CEILDIV(M, LS)*LS, LS), symmat, data, M, N);
  symmat[(M-1)*M + (M-1)] = 1.0f;
  c1 = cyc(); REPORT("corr hwacha-cc", c0, c1, (long)M*M*N);
  int bad = check_f("corr", symmat, ref, M*M, 1e-3f);
  printf("corr %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
