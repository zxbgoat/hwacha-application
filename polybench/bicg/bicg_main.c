// PolyBench bicg: q := A p, s := A^T r, two kernels. The kernel is the unmodified PolyBench/GPU
// bicg.cl (one work-item per output element, each zeroing its own accumulator first).
#include "common.h"
#define NX 64
#define NY 64
#define LS 64
void bicgKernel1_ct(long n, float *A, float *p, float *q, int nx, int ny);
void bicgKernel2_ct(long n, float *A, float *r, float *s, int nx, int ny);
static float A[NX*NY], p[NY], r[NX], q[NX], s[NY], refq[NX], refs[NY];
static void ref_kernel(void) {
  for (int i = 0; i < NX; i++) { for (int j = 0; j < NY; j++) refq[i] += A[i*NY + j] * p[j]; }
  for (int j = 0; j < NY; j++) { for (int i = 0; i < NX; i++) refs[j] += A[i*NY + j] * r[i]; }
}
int main(void) {
  for (int i = 0; i < NY; i++) p[i] = (float)(i * 3.14159265);
  for (int i = 0; i < NX; i++) { r[i] = (float)(i * 3.14159265); for (int j = 0; j < NY; j++) A[i*NY + j] = (float)(i*j) / NX; }
  for (int i = 0; i < NX; i++) { q[i] = 0; refq[i] = 0; }
  for (int i = 0; i < NY; i++) { s[i] = 0; refs[i] = 0; }
  unsigned long c0 = cyc(); ref_kernel(); unsigned long c1 = cyc();
  REPORT("bicg scalar", c0, c1, (long)NX*NY);
  c0 = cyc();
  bicgKernel1_ct(NDRANGE1(NX, LS), A, p, q, NX, NY);
  bicgKernel2_ct(NDRANGE1(NY, LS), A, r, s, NX, NY);
  c1 = cyc(); REPORT("bicg hwacha-cc", c0, c1, (long)NX*NY);
  int bad = check_f("bicg q", q, refq, NX, 1e-4f);
  bad += check_f("bicg s", s, refs, NY, 1e-4f);
  printf("bicg %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
