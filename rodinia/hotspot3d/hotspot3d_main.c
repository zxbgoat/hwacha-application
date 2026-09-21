// Rodinia hotspot3D: 3-D thermal stencil, one work-item per (x, y) column sweeping z (hotspotOpt1),
// launched as a 2-D NDRange like 3D.c; ping-pong over `iterations`, against a scalar copy
#include "common.h"
#define NX 32
#define NY 32
#define NZ 8
#define LS 8
#define ITER 2
void hotspotOpt1_ct(long n, float *p, float *tIn, float *tOut, float sdc, int nx, int ny, int nz, float ce, float cw, float cn, float cs, float ct, float cb, float cc);
static float p[NX*NY*NZ], a[NX*NY*NZ], b[NX*NY*NZ], ra[NX*NY*NZ], rb[NX*NY*NZ];
int main(void) {
  for (int i = 0; i < NX*NY*NZ; i++) { p[i] = frand(0, 1e-3f); a[i] = ra[i] = frand(320, 340); }
  float t_chip = 0.0005f, chip_height = 0.016f, chip_width = 0.016f, K_SI = 100, FACTOR_CHIP = 0.5f, SPEC_HEAT_SI = 1.75e6f, MAX_PD = 3e6f, PRECISION = 0.001f;
  float dx = chip_height / NY, dy = chip_width / NX, dz = t_chip / NZ;
  float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy;
  float Rx = dy / (2.0f * K_SI * t_chip * dx), Ry = dx / (2.0f * K_SI * t_chip * dy), Rz = dz / (K_SI * dx * dy);
  float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI), dt = PRECISION / max_slope, sdc = dt / Cap;
  float ce = sdc / Rx, cw = ce, cn = sdc / Ry, cs = cn, ct = sdc / Rz, cb = ct, cc = 1.0f - (2.0f * ce + 2.0f * cn + 3.0f * ct), amb = 80.0f;
  unsigned long c0 = cyc();
  float *src = ra, *dst = rb;
  for (int it = 0; it < ITER; it++) {
    for (int k = 0; k < NZ; k++) for (int j = 0; j < NY; j++) for (int i = 0; i < NX; i++) {
      int c = i + j * NX + k * NX * NY, xy = NX * NY;
      int W = i == 0 ? c : c - 1, E = i == NX - 1 ? c : c + 1, N = j == 0 ? c : c - NX, S = j == NY - 1 ? c : c + NX;
      float t1 = k == 0 ? src[c] : src[c - xy], t2 = src[c], t3 = k == NZ - 1 ? src[c] : src[c + xy];
      dst[c] = cc * t2 + cw * src[W] + ce * src[E] + cs * src[S] + cn * src[N] + cb * t1 + ct * t3 + sdc * p[c] + ct * amb;
    }
    float *tmp = src; src = dst; dst = tmp;
  }
  unsigned long c1 = cyc(); REPORT("hotspot3d scalar", c0, c1, (long)NX*NY*NZ*ITER);
  float *ref = src;
  c0 = cyc();
  float *hs = a, *hd = b;
  for (int it = 0; it < ITER; it++) {
    hotspotOpt1_ct(NDRANGE2(NX / LS, NY / LS, LS, LS), p, hs, hd, sdc, NX, NY, NZ, ce, cw, cn, cs, ct, cb, cc);
    float *tmp = hs; hs = hd; hd = tmp;
  }
  c1 = cyc(); REPORT("hotspot3d hwacha-cc", c0, c1, (long)NX*NY*NZ*ITER);
  int bad = check_f("temp", hs, ref, NX*NY*NZ, 1e-4f);
  printf("hotspot3d %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
