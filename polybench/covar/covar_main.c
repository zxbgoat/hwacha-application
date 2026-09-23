// PolyBench covariance (covar): column means, centring, then the M x M covariance matrix. Kernels:
// the unmodified PolyBench/GPU covariance.cl (mean, reduce, covar) in the original host's order;
// inputs and float_n = N follow PolyBenchC-4.2.1's init_array.
#include "common.h"
#define M 32
#define N 40
#define LS 64
#define LX 8
#define LY 8
void mean_kernel_ct(long n, float *mean, float *data, float float_n, int m, int n_);
void reduce_kernel_ct(long n, float *mean, float *data, int m, int n_);
void covar_kernel_ct(long n, float *symmat, float *data, int m, int n_);
static float data[N*M], mean[M], symmat[M*M], rdata[N*M], rmean[M], ref[M*M];
static const float float_n = (float)N;
static void ref_kernel(void) {
  for (int j = 0; j < M; j++) { rmean[j] = 0; for (int i = 0; i < N; i++) rmean[j] += rdata[i*M + j]; rmean[j] /= float_n; }
  for (int i = 0; i < N; i++) for (int j = 0; j < M; j++) rdata[i*M + j] -= rmean[j];
  for (int j1 = 0; j1 < M; j1++) for (int j2 = j1; j2 < M; j2++) { ref[j1*M + j2] = 0; for (int i = 0; i < N; i++) ref[j1*M + j2] += rdata[i*M + j1] * rdata[i*M + j2]; ref[j2*M + j1] = ref[j1*M + j2]; }
}
int main(void) {
  for (int i = 0; i < N; i++) for (int j = 0; j < M; j++) data[i*M + j] = rdata[i*M + j] = (float)(i*j) / M;
  unsigned long c0 = cyc(); ref_kernel(); unsigned long c1 = cyc();
  REPORT("covar scalar", c0, c1, (long)M*M*N);
  c0 = cyc();
  mean_kernel_ct(NDRANGE1(CEILDIV(M, LS)*LS, LS), mean, data, float_n, M, N);
  reduce_kernel_ct(NDRANGE2(CEILDIV(M, LX), CEILDIV(N, LY), LX, LY), mean, data, M, N);
  covar_kernel_ct(NDRANGE1(CEILDIV(M, LS)*LS, LS), symmat, data, M, N);
  c1 = cyc(); REPORT("covar hwacha-cc", c0, c1, (long)M*M*N);
  int bad = check_f("covar", symmat, ref, M*M, 1e-3f);
  printf("covar %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
