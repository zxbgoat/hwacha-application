// PolyBench atax: y := A^T (A x), two kernels. The kernel is the unmodified PolyBench/GPU atax.cl
// (atax_kernel1 computes tmp = A x with one work-item per row, atax_kernel2 accumulates y with one
// work-item per column); the reference is the sequential CPU version of the same host.
//
// The GPU kernels accumulate into tmp[i] / y[j] that the host seeds (atax.c writes the arrays to the
// device before the launch, and the CPU reference in that file adds to the same starting values), so
// tmp and y are given a fixed non-zero start here and both sides add to it.
#include "common.h"
#define NX 64
#define NY 64
#define LS 64
void atax_kernel1_ct(long n, float *A, float *x, float *tmp, int nx, int ny);
void atax_kernel2_ct(long n, float *A, float *y, float *tmp, int nx, int ny);
static float A[NX*NY], x[NY], tmp[NX], y[NY], ref[NY];
static float tmp0[NX], ystart[NY];
static void ref_kernel(void) {
  for (int i = 0; i < NX; i++) {
    tmp[i] = tmp0[i];
    for (int j = 0; j < NY; j++) tmp[i] += A[i*NY + j] * x[j];
    for (int j = 0; j < NY; j++) ref[j] += A[i*NY + j] * tmp[i];
  }
}
int main(void) {
  for (int i = 0; i < NX; i++) { x[i] = (float)(i * 3.14159265); for (int j = 0; j < NY; j++) A[i*NY + j] = (float)(i*j) / NX; }
  for (int i = 0; i < NY; i++) { y[i] = ystart[i] = 3.0f; ref[i] = ystart[i]; }
  for (int i = 0; i < NX; i++) tmp[i] = tmp0[i] = 7.0f;
  unsigned long c0 = cyc(); ref_kernel(); unsigned long c1 = cyc();
  REPORT("atax scalar", c0, c1, (long)NX*NY);
  for (int i = 0; i < NX; i++) tmp[i] = tmp0[i];
  c0 = cyc();
  atax_kernel1_ct(NDRANGE1(NX, LS), A, x, tmp, NX, NY);
  atax_kernel2_ct(NDRANGE1(NY, LS), A, y, tmp, NX, NY);
  c1 = cyc(); REPORT("atax hwacha-cc", c0, c1, (long)NX*NY);
  printf("  tmp[1] hw=%ld ref=%ld /1e3\n", (long)(tmp[1]*1e3f), (long)((tmp0[1]+4189.31f)*1e3f));
  int bad = check_f("atax", y, ref, NY, 1e-4f);
  printf("atax %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
