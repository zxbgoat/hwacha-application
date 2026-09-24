// SHOC GEMM (level1): the MAGMA-derived sgemmNN / sgemmNT kernels (64 x 16 C tile per 16 x 4
// work-group, A staged 4 elements per work-item, B through a __local 16 x 17 tile, column-major):
// C = alpha * A * op(B) + beta * C with alpha = 1, beta = -1 as GEMM.cpp, inputs uniform in
// [0.5, 2). SHOC only times these kernels; here the result is checked against a scalar reference
// (1e-4 relative, the accumulation order differs). N scaled from 256 to 128.
//
// KNOWN LIMIT: the kernels need a 64-lane work-group (the 16 x 4 tile mapping is hard-coded, the B
// tile is shared through __local memory across barriers), which on Hwacha requires <= 32 vector
// registers (maxvl = 8 * 256 / registers), but the k loop keeps 16 C accumulators, 4 A values and
// the tile addresses live: hwacha-cc allocates 65 vector registers (maxvl 24) and its spiller cannot
// evict values that stay live across the loop (-vregs 32 ends in "nothing to spill"). The group is
// therefore split over three stripmines and the barrier-separated tile loads see the wrong lanes;
// the host reports the mismatch and hwacha_vl_short. Kept as a documented FAIL.
#include "common.h"
#define N 128
#define LX 16
#define LY 4
typedef void gemm_fn(long n, const float *A, int lda, const float *B, int ldb, float *C, int ldc, int k, float alpha, float beta);
gemm_fn sgemmNN_ct, sgemmNT_ct;
static float A[N*N], B[N*N], C[N*N], C0[N*N], ref[N*N];
static int run(const char *name, gemm_fn *f, int nt) {
  const float alpha = 1, beta = -1;
  for (int i = 0; i < N*N; i++) C[i] = C0[i];
  unsigned long c0 = cyc();
  for (int j = 0; j < N; j++) for (int i = 0; i < N; i++) { float acc = 0; for (int l = 0; l < N; l++) acc += A[i + l*N] * (nt ? B[j + l*N] : B[l + j*N]); ref[i + j*N] = alpha * acc + beta * C0[i + j*N]; }
  unsigned long c1 = cyc(); unsigned long sc = c1 - c0;
  c0 = cyc(); f(NDRANGE2(N / 64, N / 16, LX, LY), A, N, B, N, C, N, N, alpha, beta); c1 = cyc();
  printf("gemm %s %dx%dx%d: scalar %lu cycles, hwacha-cc %lu cycles\n", name, N, N, N, sc, c1 - c0);
  return check_f("  C", C, ref, N*N, 1e-4f);
}
int main(void) {
  for (int i = 0; i < N*N; i++) { A[i] = 0.5f + frand(0, 1.5f); B[i] = 0.5f + frand(0, 1.5f); C0[i] = frand(0, 1); }
  int bad = run("sgemmNN", sgemmNN_ct, 0); bad += run("sgemmNT", sgemmNT_ct, 1);
  printf("gemm %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
