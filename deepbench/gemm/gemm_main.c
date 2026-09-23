// DeepBench GEMM (gemm_bench, cublasSgemm semantics, column-major): several (m, n, k, a_t, b_t) shapes
// of the training set (1760 x 16 x 1760 NN / TN, 1760 x 7133 x 1760 NT, 5124 x 9124 x 1760 ...)
// and the inference server set (35 x 700 x 2048 ...), scaled down by 16-32 for Spike. Kernels:
// gemm.cl (gemm_nn / gemm_tn / gemm_nt, one work-item per C element); reference: plain loops in the
// same order. alpha = 1, beta = 0 as gemm_bench.
#include "common.h"
#define MAXM 128
#define MAXN 128
#define MAXK 128
#define LX 8
#define LY 8
typedef void gemm_fn(long n, float *A, float *B, float *C, float alpha, float beta, int m, int n_, int k, int lda, int ldb, int ldc);
gemm_fn gemm_nn_ct, gemm_tn_ct, gemm_nt_ct;
static float A[MAXM*MAXK], B[MAXK*MAXN], C[MAXM*MAXN], ref[MAXM*MAXN];
static const float alpha = 1.0f, beta = 0.0f;
static struct { int m, n, k, a_t, b_t; const char *from; } shapes[] = {
  {110, 16, 110, 0, 0, "training 1760x16x1760 NN /16"}, {128, 32, 128, 0, 0, "training 2048x32x2048 NN /16"},
  {110, 16, 110, 1, 0, "training 1760x16x1760 TN /16"}, {110, 111, 110, 0, 1, "training 1760x7133x1760 NT /16,/64"},
  {80, 71, 55, 0, 0, "training 5124x9124x1760 /64,/128,/32"}, {35, 44, 128, 0, 0, "inference server 35x700x2048 /16"},
  {96, 1, 80, 0, 0, "inference device 3072x1x1024 /32"} };
static int run(int s) {
  int m = shapes[s].m, n = shapes[s].n, k = shapes[s].k, a_t = shapes[s].a_t, b_t = shapes[s].b_t;
  int lda = a_t ? k : m, ldb = b_t ? n : k, ldc = m;
  for (int i = 0; i < (a_t ? k*m : m*k); i++) A[i] = frand(-1, 1);
  for (int i = 0; i < (b_t ? n*k : k*n); i++) B[i] = frand(-1, 1);
  for (int i = 0; i < m*n; i++) C[i] = ref[i] = 0;
  unsigned long c0 = cyc();
  for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) { float acc = 0; for (int l = 0; l < k; l++) acc += (a_t ? A[l + i*lda] : A[i + l*lda]) * (b_t ? B[j + l*ldb] : B[l + j*ldb]); ref[i + j*ldc] = alpha * acc + beta * ref[i + j*ldc]; }
  unsigned long c1 = cyc(); unsigned long sc = c1 - c0;
  gemm_fn *f = a_t ? gemm_tn_ct : b_t ? gemm_nt_ct : gemm_nn_ct;
  c0 = cyc(); f(NDRANGE2(CEILDIV(m, LX), CEILDIV(n, LY), LX, LY), A, B, C, alpha, beta, m, n, k, lda, ldb, ldc); c1 = cyc();
  printf("gemm %dx%dx%d %s%s (%s): scalar %lu cycles, hwacha-cc %lu cycles\n", m, n, k, a_t ? "T" : "N", b_t ? "T" : "N", shapes[s].from, sc, c1 - c0);
  return check_f("  ", C, ref, m*n, 1e-4f);
}
int main(void) {
  int bad = 0; unsigned long c0 = cyc();
  for (int s = 0; s < (int)(sizeof shapes / sizeof shapes[0]); s++) bad += run(s);
  unsigned long c1 = cyc(); REPORT("gemm hwacha-cc+scalar", c0, c1, 7);
  printf("gemm %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
