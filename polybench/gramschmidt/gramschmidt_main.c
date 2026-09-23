// PolyBench gramschmidt: modified Gram-Schmidt QR, one column k at a time. Kernel: the unmodified
// PolyBench/GPU gramschmidt.cl (kernel1 computes R[k][k], kernel2 Q[:,k], kernel3 R[k][j] and the
// update of A[:,j] for j > k), with k iterated on the host as the original host does. The reference
// is PolyBenchC-4.2.1's kernel_gramschmidt (same order of operations).
#include "common.h"
#define M 48
#define N 48
#define LS 64
void gramschmidt_kernel1_ct(long n, float *a, float *r, float *q, int k, int m, int n_);
void gramschmidt_kernel2_ct(long n, float *a, float *r, float *q, int k, int m, int n_);
void gramschmidt_kernel3_ct(long n, float *a, float *r, float *q, int k, int m, int n_);
static float A[M*N], R[N*N], Q[M*N], refa[M*N], refr[N*N], refq[M*N];
static void ref_kernel(void) {
  for (int k = 0; k < N; k++) {
    float nrm = 0;
    for (int i = 0; i < M; i++) nrm += refa[i*N + k] * refa[i*N + k];
    refr[k*N + k] = sqrtf(nrm);
    for (int i = 0; i < M; i++) refq[i*N + k] = refa[i*N + k] / refr[k*N + k];
    for (int j = k + 1; j < N; j++) {
      refr[k*N + j] = 0;
      for (int i = 0; i < M; i++) refr[k*N + j] += refq[i*N + k] * refa[i*N + j];
      for (int i = 0; i < M; i++) refa[i*N + j] = refa[i*N + j] - refq[i*N + k] * refr[k*N + j];
    }
  }
}
int main(void) {
  for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) { A[i*N + j] = refa[i*N + j] = (float)(((i*j) % M) / M) * 100 + 10; Q[i*N + j] = refq[i*N + j] = 0; }
  for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) R[i*N + j] = refr[i*N + j] = 0;
  unsigned long c0 = cyc(); ref_kernel(); unsigned long c1 = cyc();
  REPORT("gramschmidt scalar", c0, c1, (long)M*N*N);
  c0 = cyc();
  for (int k = 0; k < N; k++) {
    gramschmidt_kernel1_ct(NDRANGE1(CEILDIV(M, LS)*LS, LS), A, R, Q, k, M, N);
    gramschmidt_kernel2_ct(NDRANGE1(CEILDIV(M, LS)*LS, LS), A, R, Q, k, M, N);
    if (k < N - 1) gramschmidt_kernel3_ct(NDRANGE1(CEILDIV(N, LS)*LS, LS), A, R, Q, k, M, N);
  }
  c1 = cyc(); REPORT("gramschmidt hwacha-cc", c0, c1, (long)M*N*N);
  int bad = check_f("gramschmidt A", A, refa, M*N, 1e-4f);
  bad += check_f("gramschmidt R", R, refr, N*N, 1e-4f);
  bad += check_f("gramschmidt Q", Q, refq, M*N, 1e-4f);
  printf("gramschmidt %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
