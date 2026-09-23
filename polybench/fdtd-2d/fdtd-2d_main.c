// PolyBench fdtd-2d: TMAX steps of the 2-D FDTD update (ey, ex, then hz). Kernels: the unmodified
// PolyBench/GPU fdtd2d.cl, three 2-D launches per step; reference: runFdtd of that host.
#include "common.h"
#define TMAX 4
#define NX 48
#define NY 48
#define LX 8
#define LY 8
void fdtd_kernel1_ct(long n, float *_fict_, float *ex, float *ey, float *hz, int t, int nx, int ny);
void fdtd_kernel2_ct(long n, float *ex, float *ey, float *hz, int nx, int ny);
void fdtd_kernel3_ct(long n, float *ex, float *ey, float *hz, int nx, int ny);
static float fict[TMAX], ex[NX*NY], ey[NX*NY], hz[NX*NY], rex[NX*NY], rey[NX*NY], rhz[NX*NY];
static void ref_kernel(void) {
  for (int t = 0; t < TMAX; t++) {
    for (int j = 0; j < NY; j++) rey[j] = fict[t];
    for (int i = 1; i < NX; i++) for (int j = 0; j < NY; j++) rey[i*NY + j] = rey[i*NY + j] - 0.5f*(rhz[i*NY + j] - rhz[(i-1)*NY + j]);
    for (int i = 0; i < NX; i++) for (int j = 1; j < NY; j++) rex[i*NY + j] = rex[i*NY + j] - 0.5f*(rhz[i*NY + j] - rhz[i*NY + (j-1)]);
    for (int i = 0; i < NX - 1; i++) for (int j = 0; j < NY - 1; j++) rhz[i*NY + j] = rhz[i*NY + j] - 0.7f*(rex[i*NY + (j+1)] - rex[i*NY + j] + rey[(i+1)*NY + j] - rey[i*NY + j]);
  }
}
int main(void) {
  for (int i = 0; i < TMAX; i++) fict[i] = (float)i;
  for (int i = 0; i < NX; i++) for (int j = 0; j < NY; j++) {
    ex[i*NY + j] = rex[i*NY + j] = ((float)(i*(j+1) + 1)) / NX; ey[i*NY + j] = rey[i*NY + j] = ((float)((i-1)*(j+2) + 2)) / NX; hz[i*NY + j] = rhz[i*NY + j] = ((float)((i-9)*(j+4) + 3)) / NX; }
  unsigned long c0 = cyc(); ref_kernel(); unsigned long c1 = cyc();
  REPORT("fdtd-2d scalar", c0, c1, (long)NX*NY*TMAX);
  c0 = cyc();
  for (int t = 0; t < TMAX; t++) {
    fdtd_kernel1_ct(NDRANGE2(CEILDIV(NY, LX), CEILDIV(NX, LY), LX, LY), fict, ex, ey, hz, t, NX, NY);
    fdtd_kernel2_ct(NDRANGE2(CEILDIV(NY, LX), CEILDIV(NX, LY), LX, LY), ex, ey, hz, NX, NY);
    fdtd_kernel3_ct(NDRANGE2(CEILDIV(NY, LX), CEILDIV(NX, LY), LX, LY), ex, ey, hz, NX, NY);
  }
  c1 = cyc(); REPORT("fdtd-2d hwacha-cc", c0, c1, (long)NX*NY*TMAX);
  int bad = check_f("fdtd-2d ex", ex, rex, NX*NY, 1e-4f);
  bad += check_f("fdtd-2d ey", ey, rey, NX*NY, 1e-4f);
  bad += check_f("fdtd-2d hz", hz, rhz, NX*NY, 1e-4f);
  printf("fdtd-2d %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
