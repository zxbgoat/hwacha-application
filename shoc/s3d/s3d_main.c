// SHOC S3D (level2): one evaluation of the 22-species / 206-reaction chemical kinetics right-hand
// side (species production rates WDOT) with SHOC's 27 unmodified kernels, launched in S3D.cpp's order
// (phase 1: gr_base, ratt, rdsmh, ratt2..ratt10, ratx, ratxb, ratx2, ratx4; phase 2: qssa, qssab,
// qssa2, rdwdot, rdwdot2, rdwdot3, rdwdot6..rdwdot10), one work-item per grid point. Inputs as
// S3D.cpp: p = 1.0132e6, t = 1000, y = 0.218 / 0.064 / 0.718 for species 3 / 14 / 21, molwt = 1,
// tconv = pconv = rateconv = 1. N_GP (grid points per launch) is baked into the kernels (-DN_GP=64).
// SHOC does not verify S3D; here the same 27 files compiled as C (s3d_ref.c) are the reference:
// relative error per element within 1e-2 (hwacha-cc's exp / log expansions, 253 exp per point, long
// dependency chains through rates spanning many decades).
#include "common.h"
#include <float.h>
#define N 64
#define LS 64
#define NSP 22
#define RFS 206
#define EGS 32
#define RKS 21
#define AS 121
void gr_base_ct(long n, float * P, float * T, float * Y, float * C, float TCONV, float PCONV);
void ratt_kernel_ct(long n, float * T, float * RF, float TCONV);
void rdsmh_kernel_ct(long n, float * T, float * EG, float TCONV);
void ratt2_kernel_ct(long n, float * T, float * RF, float * RB, float * EG, float TCONV);
void ratt3_kernel_ct(long n, float * T, float * RF, float * RB, float * EG, float TCONV);
void ratt4_kernel_ct(long n, float * T, float * RF, float * RB, float * EG, float TCONV);
void ratt5_kernel_ct(long n, float * T, float * RF, float * RB, float * EG, float TCONV);
void ratt6_kernel_ct(long n, float * T, float * RF, float * RB, float * EG, float TCONV);
void ratt7_kernel_ct(long n, float * T, float * RF, float * RB, float * EG, float TCONV);
void ratt8_kernel_ct(long n, float * T, float * RF, float * RB, float * EG, float TCONV);
void ratt9_kernel_ct(long n, float * T, float * RF, float * RB, float * EG, float TCONV);
void ratt10_kernel_ct(long n, float * T, float * RKLOW, float TCONV);
void ratx_kernel_ct(long n, float * T, float * C, float * RF, float * RB, float * RKLOW, float TCONV);
void ratxb_kernel_ct(long n, float * T, float * C, float * RF, float * RB, float * RKLOW, float TCONV);
void ratx2_kernel_ct(long n, float * C, float * RF);
void ratx4_kernel_ct(long n, float * C, float * RB);
void qssa_kernel_ct(long n, float * RF, float * RB, float * A);
void qssab_kernel_ct(long n, float * RF, float * RB, float * A);
void qssa2_kernel_ct(long n, float * RF, float * RB, float * A);
void rdwdot_kernel_ct(long n, float * RKF, float * RKR, float * WDOT, float rateconv, float * molwt);
void rdwdot2_kernel_ct(long n, float * RKF, float * RKR, float * WDOT, float rateconv, float * molwt);
void rdwdot3_kernel_ct(long n, float * RKF, float * RKR, float * WDOT, float rateconv, float * molwt);
void rdwdot6_kernel_ct(long n, float * RKF, float * RKR, float * WDOT, float rateconv, float * molwt);
void rdwdot7_kernel_ct(long n, float * RKF, float * RKR, float * WDOT, float rateconv, float * molwt);
void rdwdot8_kernel_ct(long n, float * RKF, float * RKR, float * WDOT, float rateconv, float * molwt);
void rdwdot9_kernel_ct(long n, float * RKF, float * RKR, float * WDOT, float rateconv, float * molwt);
void rdwdot10_kernel_ct(long n, float * RKF, float * RKR, float * WDOT, float rateconv, float * molwt);

extern int s3d_gid;
void gr_base(float * P, float * T, float * Y, float * C, float TCONV, float PCONV);
void ratt_kernel(float * T, float * RF, float TCONV);
void rdsmh_kernel(float * T, float * EG, float TCONV);
void ratt2_kernel(float * T, float * RF, float * RB, float * EG, float TCONV);
void ratt3_kernel(float * T, float * RF, float * RB, float * EG, float TCONV);
void ratt4_kernel(float * T, float * RF, float * RB, float * EG, float TCONV);
void ratt5_kernel(float * T, float * RF, float * RB, float * EG, float TCONV);
void ratt6_kernel(float * T, float * RF, float * RB, float * EG, float TCONV);
void ratt7_kernel(float * T, float * RF, float * RB, float * EG, float TCONV);
void ratt8_kernel(float * T, float * RF, float * RB, float * EG, float TCONV);
void ratt9_kernel(float * T, float * RF, float * RB, float * EG, float TCONV);
void ratt10_kernel(float * T, float * RKLOW, float TCONV);
void ratx_kernel(float * T, float * C, float * RF, float * RB, float * RKLOW, float TCONV);
void ratxb_kernel(float * T, float * C, float * RF, float * RB, float * RKLOW, float TCONV);
void ratx2_kernel(float * C, float * RF);
void ratx4_kernel(float * C, float * RB);
void qssa_kernel(float * RF, float * RB, float * A);
void qssab_kernel(float * RF, float * RB, float * A);
void qssa2_kernel(float * RF, float * RB, float * A);
void rdwdot_kernel(float * RKF, float * RKR, float * WDOT, float rateconv, float * molwt);
void rdwdot2_kernel(float * RKF, float * RKR, float * WDOT, float rateconv, float * molwt);
void rdwdot3_kernel(float * RKF, float * RKR, float * WDOT, float rateconv, float * molwt);
void rdwdot6_kernel(float * RKF, float * RKR, float * WDOT, float rateconv, float * molwt);
void rdwdot7_kernel(float * RKF, float * RKR, float * WDOT, float rateconv, float * molwt);
void rdwdot8_kernel(float * RKF, float * RKR, float * WDOT, float rateconv, float * molwt);
void rdwdot9_kernel(float * RKF, float * RKR, float * WDOT, float rateconv, float * molwt);
void rdwdot10_kernel(float * RKF, float * RKR, float * WDOT, float rateconv, float * molwt);

static float t[N], p[N], y[NSP*N], c[NSP*N], eg[EGS*N], rf[RFS*N], rb[RFS*N], rklow[RKS*N], abuf[(AS+2)*N], wdot[NSP*N], molwt[NSP];
static float rc[NSP*N], reg[EGS*N], rrf[RFS*N], rrb[RFS*N], rrklow[RKS*N], rabuf[(AS+2)*N], rwdot[NSP*N];
static float *a = abuf + N, *ra = rabuf + N;   // A(0,0) indexes one row before the buffer (idx2 subtracts 1): give it room
static int cmp(const char *tag, const float *hw, const float *ref, int n, float rtol) {
  int bad = 0; float mr = 0; for (int i = 0; i < n; i++) { float d = fabsf(hw[i] - ref[i]), r = d / (fabsf(ref[i]) + 1e-30f); if (d > rtol * fabsf(ref[i]) + 1e-20f) bad++; if (r > mr && fabsf(ref[i]) > 1e-20f) mr = r; }
  printf("  %s: %d of %d above %ld/1e6 relative, max relative error %ld/1e6\n", tag, bad, n, (long)(rtol*1e6f), (long)(mr*1e6f)); return bad; }
int main(void) {
  const float tconv = 1.0f, pconv = 1.0f, rateconv = 1.0f;
  for (int i = 0; i < N; i++) { p[i] = 1.0132e6f; t[i] = 1000.0f; }
  for (int j = 0; j < NSP; j++) for (int i = 0; i < N; i++) y[j*N + i] = j == 14 ? 0.064f : j == 3 ? 0.218f : j == 21 ? 0.718f : 0.0f;
  for (int i = 0; i < NSP; i++) molwt[i] = 1.0f;
  float *rp = p, *rt = t, *ry = y, *rmolwt = molwt;
  unsigned long c0 = cyc();
  for (s3d_gid = 0; s3d_gid < N; s3d_gid++) gr_base(rp, rt, ry, rc, tconv, pconv);
  for (s3d_gid = 0; s3d_gid < N; s3d_gid++) ratt_kernel(rt, rrf, tconv);
  for (s3d_gid = 0; s3d_gid < N; s3d_gid++) rdsmh_kernel(rt, reg, tconv);
  for (s3d_gid = 0; s3d_gid < N; s3d_gid++) ratt2_kernel(rt, rrf, rrb, reg, tconv);
  for (s3d_gid = 0; s3d_gid < N; s3d_gid++) ratt3_kernel(rt, rrf, rrb, reg, tconv);
  for (s3d_gid = 0; s3d_gid < N; s3d_gid++) ratt4_kernel(rt, rrf, rrb, reg, tconv);
  for (s3d_gid = 0; s3d_gid < N; s3d_gid++) ratt5_kernel(rt, rrf, rrb, reg, tconv);
  for (s3d_gid = 0; s3d_gid < N; s3d_gid++) ratt6_kernel(rt, rrf, rrb, reg, tconv);
  for (s3d_gid = 0; s3d_gid < N; s3d_gid++) ratt7_kernel(rt, rrf, rrb, reg, tconv);
  for (s3d_gid = 0; s3d_gid < N; s3d_gid++) ratt8_kernel(rt, rrf, rrb, reg, tconv);
  for (s3d_gid = 0; s3d_gid < N; s3d_gid++) ratt9_kernel(rt, rrf, rrb, reg, tconv);
  for (s3d_gid = 0; s3d_gid < N; s3d_gid++) ratt10_kernel(rt, rrklow, tconv);
  for (s3d_gid = 0; s3d_gid < N; s3d_gid++) ratx_kernel(rt, rc, rrf, rrb, rrklow, tconv);
  for (s3d_gid = 0; s3d_gid < N; s3d_gid++) ratxb_kernel(rt, rc, rrf, rrb, rrklow, tconv);
  for (s3d_gid = 0; s3d_gid < N; s3d_gid++) ratx2_kernel(rc, rrf);
  for (s3d_gid = 0; s3d_gid < N; s3d_gid++) ratx4_kernel(rc, rrb);
  for (s3d_gid = 0; s3d_gid < N; s3d_gid++) qssa_kernel(rrf, rrb, ra);
  for (s3d_gid = 0; s3d_gid < N; s3d_gid++) qssab_kernel(rrf, rrb, ra);
  for (s3d_gid = 0; s3d_gid < N; s3d_gid++) qssa2_kernel(rrf, rrb, ra);
  for (s3d_gid = 0; s3d_gid < N; s3d_gid++) rdwdot_kernel(rrf, rrb, rwdot, rateconv, rmolwt);
  for (s3d_gid = 0; s3d_gid < N; s3d_gid++) rdwdot2_kernel(rrf, rrb, rwdot, rateconv, rmolwt);
  for (s3d_gid = 0; s3d_gid < N; s3d_gid++) rdwdot3_kernel(rrf, rrb, rwdot, rateconv, rmolwt);
  for (s3d_gid = 0; s3d_gid < N; s3d_gid++) rdwdot6_kernel(rrf, rrb, rwdot, rateconv, rmolwt);
  for (s3d_gid = 0; s3d_gid < N; s3d_gid++) rdwdot7_kernel(rrf, rrb, rwdot, rateconv, rmolwt);
  for (s3d_gid = 0; s3d_gid < N; s3d_gid++) rdwdot8_kernel(rrf, rrb, rwdot, rateconv, rmolwt);
  for (s3d_gid = 0; s3d_gid < N; s3d_gid++) rdwdot9_kernel(rrf, rrb, rwdot, rateconv, rmolwt);
  for (s3d_gid = 0; s3d_gid < N; s3d_gid++) rdwdot10_kernel(rrf, rrb, rwdot, rateconv, rmolwt);
  unsigned long c1 = cyc(); REPORT("s3d scalar", c0, c1, N);
  c0 = cyc();
  gr_base_ct(NDRANGE1(N, LS), p, t, y, c, tconv, pconv);
  ratt_kernel_ct(NDRANGE1(N, LS), t, rf, tconv);
  rdsmh_kernel_ct(NDRANGE1(N, LS), t, eg, tconv);
  ratt2_kernel_ct(NDRANGE1(N, LS), t, rf, rb, eg, tconv);
  ratt3_kernel_ct(NDRANGE1(N, LS), t, rf, rb, eg, tconv);
  ratt4_kernel_ct(NDRANGE1(N, LS), t, rf, rb, eg, tconv);
  ratt5_kernel_ct(NDRANGE1(N, LS), t, rf, rb, eg, tconv);
  ratt6_kernel_ct(NDRANGE1(N, LS), t, rf, rb, eg, tconv);
  ratt7_kernel_ct(NDRANGE1(N, LS), t, rf, rb, eg, tconv);
  ratt8_kernel_ct(NDRANGE1(N, LS), t, rf, rb, eg, tconv);
  ratt9_kernel_ct(NDRANGE1(N, LS), t, rf, rb, eg, tconv);
  ratt10_kernel_ct(NDRANGE1(N, LS), t, rklow, tconv);
  ratx_kernel_ct(NDRANGE1(N, LS), t, c, rf, rb, rklow, tconv);
  ratxb_kernel_ct(NDRANGE1(N, LS), t, c, rf, rb, rklow, tconv);
  ratx2_kernel_ct(NDRANGE1(N, LS), c, rf);
  ratx4_kernel_ct(NDRANGE1(N, LS), c, rb);
  qssa_kernel_ct(NDRANGE1(N, LS), rf, rb, a);
  qssab_kernel_ct(NDRANGE1(N, LS), rf, rb, a);
  qssa2_kernel_ct(NDRANGE1(N, LS), rf, rb, a);
  rdwdot_kernel_ct(NDRANGE1(N, LS), rf, rb, wdot, rateconv, molwt);
  rdwdot2_kernel_ct(NDRANGE1(N, LS), rf, rb, wdot, rateconv, molwt);
  rdwdot3_kernel_ct(NDRANGE1(N, LS), rf, rb, wdot, rateconv, molwt);
  rdwdot6_kernel_ct(NDRANGE1(N, LS), rf, rb, wdot, rateconv, molwt);
  rdwdot7_kernel_ct(NDRANGE1(N, LS), rf, rb, wdot, rateconv, molwt);
  rdwdot8_kernel_ct(NDRANGE1(N, LS), rf, rb, wdot, rateconv, molwt);
  rdwdot9_kernel_ct(NDRANGE1(N, LS), rf, rb, wdot, rateconv, molwt);
  rdwdot10_kernel_ct(NDRANGE1(N, LS), rf, rb, wdot, rateconv, molwt);
  c1 = cyc(); REPORT("s3d hwacha-cc", c0, c1, N);
  { double sw = 0, srf = 0; float wmax = 0; for (int i = 0; i < NSP*N; i++) { sw += fabsf(wdot[i]); if (fabsf(wdot[i]) > wmax) wmax = fabsf(wdot[i]); } for (int i = 0; i < RFS*N; i++) srf += fabsf(rf[i]);
    printf("  magnitudes: sum|WDOT| %ld, max|WDOT| %ld, sum|RF| %ld (x1e6, non-zero on Hwacha: %s)\n", (long)(sw*1e6), (long)(wmax*1e6), (long)(srf*1e6), sw > 0 && srf > 0 ? "yes" : "NO"); }
  int bad = cmp("C (concentrations, gr_base)", c, rc, NSP*N, 1e-3f);
  bad += cmp("EG (rdsmh)", eg, reg, EGS*N, 1e-3f);
  bad += cmp("RKLOW (ratt10)", rklow, rrklow, RKS*N, 1e-2f);
  bad += cmp("RF (forward rates)", rf, rrf, RFS*N, 1e-2f);
  bad += cmp("RB (reverse rates)", rb, rrb, RFS*N, 1e-2f);
  bad += cmp("WDOT (production rates)", wdot, rwdot, NSP*N, 1e-2f);
  printf("s3d %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
