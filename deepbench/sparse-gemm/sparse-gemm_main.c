// DeepBench sparse GEMM (sparse_bench: cusparseScsrmm, A in CSR with 90% / 95% sparsity, B dense,
// alpha = 1/k, beta = 0): shapes of the inference sets (7680 x 1 x 2560, 7680 x 1500 x 2560,
// 10752 x 4 x 3584 ...) scaled down. A is generated as DeepBench does: uniform [0,1) values,
// entries below the sparsity threshold set to zero, then converted to CSR. Kernel: sparse-gemm.cl
// (csrmm); reference: the same CSR walk in scalar code.
#include "common.h"
#define MAXM 240
#define MAXK 160
#define MAXN 48
#define LX 8
#define LY 8
void csrmm_ct(long n, float *val, int *rowptr, int *colind, float *B, float *C, float alpha, float beta, int m, int n_, int k);
static float dense[MAXM*MAXK], val[MAXM*MAXK], B[MAXK*MAXN], C[MAXM*MAXN], ref[MAXM*MAXN];
static int rowptr[MAXM+1], colind[MAXM*MAXK];
static struct { int m, n, k; float sparsity; const char *from; } shapes[] = {
  {240, 1, 80, 0.95f, "server 7680x1x2560 /32"}, {240, 47, 80, 0.95f, "server 7680x1500x2560 /32"},
  {168, 4, 112, 0.95f, "server 10752x4x3584 /64,/32"}, {240, 47, 80, 0.9f, "server 7680x1500x2560 /32, 0.9"},
  {168, 47, 112, 0.9f, "device 10752x1500x3584 /64,/32, 0.9"} };
static int run(int s) {
  int m = shapes[s].m, n = shapes[s].n, k = shapes[s].k; float sp = shapes[s].sparsity;
  int nnz = 0;
  for (int i = 0; i < m; i++) { rowptr[i] = nnz; for (int j = 0; j < k; j++) { float v = frand(0, 1); if (v < sp) v = 0; dense[i*k + j] = v; if (v != 0) { val[nnz] = v; colind[nnz] = j; nnz++; } } }
  rowptr[m] = nnz;
  for (int i = 0; i < k*n; i++) B[i] = frand(-1, 1);
  for (int i = 0; i < m*n; i++) C[i] = ref[i] = 0;
  float alpha = 1.0f / (float)k, beta = 0.0f;
  unsigned long c0 = cyc();
  for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) { float acc = 0; for (int p = rowptr[i]; p < rowptr[i+1]; p++) acc += val[p] * B[colind[p] + j*k]; ref[i + j*m] = alpha * acc + beta * ref[i + j*m]; }
  unsigned long c1 = cyc(); unsigned long sc = c1 - c0;
  c0 = cyc(); csrmm_ct(NDRANGE2(CEILDIV(m, LX), CEILDIV(n, LY), LX, LY), val, rowptr, colind, B, C, alpha, beta, m, n, k); c1 = cyc();
  printf("sparse-gemm %dx%dx%d nnz %d (%s): scalar %lu cycles, hwacha-cc %lu cycles\n", m, n, k, nnz, shapes[s].from, sc, c1 - c0);
  return check_f("  ", C, ref, m*n, 1e-4f);
}
int main(void) {
  int bad = 0; for (int s = 0; s < (int)(sizeof shapes / sizeof shapes[0]); s++) bad += run(s);
  printf("sparse-gemm %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
