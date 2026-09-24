// DeepBench GEMM, half precision (gemm_bench "half": 16F inputs / outputs, 32F compute), the same
// training / inference shapes as the gemm case. Kernels: gemm-fp16.cl (gemm_nn / tn / nt: half in,
// float accumulate, half out; gemm_nn_h: pure half arithmetic). The reference converts the halves to
// float, accumulates in float and rounds once with round-to-nearest-even (vfcvt.h.s), so the compare
// is exact; for gemm_nn_h it rounds after every fused multiply-add as vfmadd.h does.
#include "common.h"
#define MAXM 128
#define MAXN 128
#define MAXK 128
#define LX 8
#define LY 8
typedef void gemm_fn(long n, f16 *A, f16 *B, f16 *C, float alpha, float beta, int m, int n_, int k, int lda, int ldb, int ldc);
gemm_fn gemm_nn_ct, gemm_tn_ct, gemm_nt_ct;
void gemm_nn_h_ct(long n, f16 *A, f16 *B, f16 *C, int m, int n_, int k, int lda, int ldb, int ldc);
static f16 A[MAXM*MAXK], B[MAXK*MAXN], C[MAXM*MAXN]; static float Cf[MAXM*MAXN], ref[MAXM*MAXN];
static struct { int m, n, k, a_t, b_t; const char *from; } shapes[] = {
  {110, 16, 110, 0, 0, "training 1760x16x1760 NN /16"}, {128, 32, 128, 0, 0, "training 2048x32x2048 NN /16"},
  {110, 16, 110, 1, 0, "training 1760x16x1760 TN /16"}, {110, 111, 110, 0, 1, "training 1760x7133x1760 NT /16,/64"},
  {80, 71, 55, 0, 0, "training 5124x9124x1760 /64,/128,/32"}, {35, 44, 128, 0, 0, "inference server 35x700x2048 /16"} };
static int run(int s, int hmath) {
  int m = shapes[s].m, n = shapes[s].n, k = shapes[s].k, a_t = shapes[s].a_t, b_t = shapes[s].b_t;
  int lda = a_t ? k : m, ldb = b_t ? n : k, ldc = m;
  for (int i = 0; i < (a_t ? k*m : m*k); i++) A[i] = f32_to_f16(frand(-1, 1));
  for (int i = 0; i < (b_t ? n*k : k*n); i++) B[i] = f32_to_f16(frand(-1, 1));
  for (int i = 0; i < m*n; i++) C[i] = 0;
  unsigned long c0 = cyc();
  for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) {
    float acc = 0;
    for (int l = 0; l < k; l++) { float a = f16_to_f32(a_t ? A[l + i*lda] : A[i + l*lda]), b = f16_to_f32(b_t ? B[j + l*ldb] : B[l + j*ldb]); acc = hmath ? h_round(a * b + acc) : acc + a * b; }
    ref[i + j*ldc] = h_round(acc); }
  unsigned long c1 = cyc(); unsigned long sc = c1 - c0;
  c0 = cyc();
  if (hmath) gemm_nn_h_ct(NDRANGE2(CEILDIV(m, LX), CEILDIV(n, LY), LX, LY), A, B, C, m, n, k, lda, ldb, ldc);
  else (a_t ? gemm_tn_ct : b_t ? gemm_nt_ct : gemm_nn_ct)(NDRANGE2(CEILDIV(m, LX), CEILDIV(n, LY), LX, LY), A, B, C, 1.0f, 0.0f, m, n, k, lda, ldb, ldc);
  c1 = cyc();
  for (int i = 0; i < m*n; i++) Cf[i] = f16_to_f32(C[i]);
  printf("gemm-fp16 %dx%dx%d %s%s%s (%s): scalar %lu cycles, hwacha-cc %lu cycles\n", m, n, k, a_t ? "T" : "N", b_t ? "T" : "N", hmath ? " half math" : "", shapes[s].from, sc, c1 - c0);
  return check_f("  ", Cf, ref, m*n, 1e-5f);
}
int main(void) {
  int bad = 0; for (int s = 0; s < (int)(sizeof shapes / sizeof shapes[0]); s++) bad += run(s, 0);
  bad += run(0, 1); bad += run(4, 1);
  printf("gemm-fp16 %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
