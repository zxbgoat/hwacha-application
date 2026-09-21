// Rodinia leukocyte, detection stage (find_ellipse_kernel.cl, buffer variant): GICOV_kernel scores
// every pixel by the best of NCIRCLES circles (NPOINTS samples each, gradient projected on the
// circle normal, variance-normalised mean), dilate_kernel then takes the max over a strel window;
// one work-item per pixel as find_ellipse_opencl.c launches them. Reference = scalar port.
// (The tracking stage, track_ellipse_kernel, uses atan(), which hwacha-cc has no expansion for.)
#include "common.h"
#define NPOINTS 150
#define NCIRCLES 7
#define MAX_RAD 20
#define MIN_RAD 8
#define W 24
#define H 24
#define GM (H + 2 * MAX_RAD + 4)     // gradient image rows (with the MAX_RAD+2 border)
#define GN (W + 2 * MAX_RAD + 4)
#define STREL 25
void GICOV_kernel_ct(long n, int grad_m, float *grad_x, float *grad_y, float *c_sin, float *c_cos, int *c_tX, int *c_tY, float *gicov, int width, int height);
void dilate_kernel_ct(long n, int img_m, int img_n, int strel_m, int strel_n, float *c_strel, float *img, float *dilated);
static float gx[GM*GN], gy[GM*GN], gicov[GM*GN], rgicov[GM*GN], dil[GM*GN], rdil[GM*GN], csin[NPOINTS], ccos[NPOINTS], strel[STREL*STREL];
static int tX[NCIRCLES*NPOINTS], tY[NCIRCLES*NPOINTS];
int main(void) {
  for (int i = 0; i < GM*GN; i++) { gx[i] = frand(-1, 1); gy[i] = frand(-1, 1); }
  for (int n = 0; n < NPOINTS; n++) { float th = (float)n * 2.0f * 3.14159265f / NPOINTS; csin[n] = sinf(th); ccos[n] = cosf(th);
    for (int k = 0; k < NCIRCLES; k++) { float rad = MIN_RAD + 2.0f * k; tX[k*NPOINTS + n] = (int)(ccos[n] * rad); tY[k*NPOINTS + n] = (int)(csin[n] * rad); } }
  for (int i = 0; i < STREL; i++) for (int j = 0; j < STREL; j++) { float di = i - STREL/2, dj = j - STREL/2; strel[i*STREL + j] = (di*di + dj*dj <= (STREL/2)*(STREL/2)) ? 1.0f : 0.0f; }
  unsigned long c0 = cyc();
  for (int gid = 0; gid < W*H; gid++) { int i = gid / W + MAX_RAD + 2, j = gid % W + MAX_RAD + 2; float mx = 0.f;
    for (int k = 0; k < NCIRCLES; k++) { float sum = 0.f, M2 = 0.f, mean = 0.f;
      for (int n = 0; n < NPOINTS; n++) { int y = j + tY[k*NPOINTS + n], x = i + tX[k*NPOINTS + n]; int addr = x * GM + y; float p = gx[addr] * ccos[n] + gy[addr] * csin[n];
        sum += p; float delta = p - mean; mean = mean + (delta / (float)(n + 1)); M2 = M2 + (delta * (p - mean)); }
      mean = sum / (float)NPOINTS; float var = M2 / (float)(NPOINTS - 1); if (((mean * mean) / var) > mx) mx = (mean * mean) / var; }
    rgicov[i * GM + j] = mx; }
  for (int t = 0; t < GM*GN; t++) { int i = t % GM, j = t / GM; if (j > GN) continue; float mx = 0.0f;
    if (j < GN) { for (int ei = 0; ei < STREL; ei++) { int y = i - STREL/2 + ei; if (y >= 0 && y < GM) for (int ej = 0; ej < STREL; ej++) { int x = j - STREL/2 + ej; if (x >= 0 && x < GN && strel[ei*STREL + ej] != 0) { float v = rgicov[y + x*GM]; if (v > mx) mx = v; } } } }
    rdil[i*GN + j] = mx; }   // the kernel stores row-major with img_n as the stride
  unsigned long c1 = cyc(); REPORT("leukocyte scalar", c0, c1, (long)W*H);
  c0 = cyc();
  GICOV_kernel_ct(NDRANGE1(W*H, 0), GM, gx, gy, csin, ccos, tX, tY, gicov, W, H);
  dilate_kernel_ct(NDRANGE1(GM*GN, 0), GM, GN, STREL, STREL, strel, gicov, dil);
  c1 = cyc(); REPORT("leukocyte hwacha-cc", c0, c1, (long)W*H);
  int bad = check_f("gicov", gicov, rgicov, GM*GN, 1e-3f) + check_f("dilated", dil, rdil, GM*GN, 1e-3f);
  printf("leukocyte %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
