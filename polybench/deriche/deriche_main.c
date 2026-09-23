// PolyBench deriche (4.2.1): Deriche recursive Gaussian edge filter, four IIR passes (rows forward
// and backward, columns forward and backward) with a combination step after each pair. Kernels:
// deriche.cl, one work-item per row / column for the recurrences, one per pixel for the
// combinations; the coefficients are computed on the host as kernel_deriche does (expf, powf).
#include "common.h"
#define W 32
#define H 32
#define LS 64
#define LX 8
#define LY 8
void deriche_kernel1_ct(long n, float *imgIn, float *y1, float a1, float a2, float b1, float b2, int w, int h);
void deriche_kernel2_ct(long n, float *imgIn, float *y2, float a3, float a4, float b1, float b2, int w, int h);
void deriche_kernel3_ct(long n, float *imgOut, float *y1, float *y2, float c, int w, int h);
void deriche_kernel4_ct(long n, float *imgOut, float *y1, float a5, float a6, float b1, float b2, int w, int h);
void deriche_kernel5_ct(long n, float *imgOut, float *y2, float a7, float a8, float b1, float b2, int w, int h);
static float imgIn[W*H], imgOut[W*H], Y1[W*H], Y2[W*H], ref[W*H], ry1[W*H], ry2[W*H];
static const float alpha = 0.25f;
static float k, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, c1, c2;
static void ref_kernel(void) {
  for (int i = 0; i < W; i++) { float ym1 = 0, ym2 = 0, xm1 = 0; for (int j = 0; j < H; j++) { ry1[i*H + j] = a1*imgIn[i*H + j] + a2*xm1 + b1*ym1 + b2*ym2; xm1 = imgIn[i*H + j]; ym2 = ym1; ym1 = ry1[i*H + j]; } }
  for (int i = 0; i < W; i++) { float yp1 = 0, yp2 = 0, xp1 = 0, xp2 = 0; for (int j = H-1; j >= 0; j--) { ry2[i*H + j] = a3*xp1 + a4*xp2 + b1*yp1 + b2*yp2; xp2 = xp1; xp1 = imgIn[i*H + j]; yp2 = yp1; yp1 = ry2[i*H + j]; } }
  for (int i = 0; i < W; i++) for (int j = 0; j < H; j++) ref[i*H + j] = c1 * (ry1[i*H + j] + ry2[i*H + j]);
  for (int j = 0; j < H; j++) { float tm1 = 0, ym1 = 0, ym2 = 0; for (int i = 0; i < W; i++) { ry1[i*H + j] = a5*ref[i*H + j] + a6*tm1 + b1*ym1 + b2*ym2; tm1 = ref[i*H + j]; ym2 = ym1; ym1 = ry1[i*H + j]; } }
  for (int j = 0; j < H; j++) { float tp1 = 0, tp2 = 0, yp1 = 0, yp2 = 0; for (int i = W-1; i >= 0; i--) { ry2[i*H + j] = a7*tp1 + a8*tp2 + b1*yp1 + b2*yp2; tp2 = tp1; tp1 = ref[i*H + j]; yp2 = yp1; yp1 = ry2[i*H + j]; } }
  for (int i = 0; i < W; i++) for (int j = 0; j < H; j++) ref[i*H + j] = c2 * (ry1[i*H + j] + ry2[i*H + j]);
}
int main(void) {
  for (int i = 0; i < W; i++) for (int j = 0; j < H; j++) imgIn[i*H + j] = (float)((313*i + 991*j) % 65536) / 65535.0f;
  k = (1.0f - expf(-alpha)) * (1.0f - expf(-alpha)) / (1.0f + 2.0f*alpha*expf(-alpha) - expf(2.0f*alpha));
  a1 = a5 = k; a2 = a6 = k*expf(-alpha)*(alpha - 1.0f); a3 = a7 = k*expf(-alpha)*(alpha + 1.0f); a4 = a8 = -k*expf(-2.0f*alpha);
  b1 = powf(2.0f, -alpha); b2 = -expf(-2.0f*alpha); c1 = c2 = 1;
  unsigned long c0 = cyc(); ref_kernel(); unsigned long cc1 = cyc();
  REPORT("deriche scalar", c0, cc1, (long)W*H);
  c0 = cyc();
  deriche_kernel1_ct(NDRANGE1(CEILDIV(W, LS)*LS, LS), imgIn, Y1, a1, a2, b1, b2, W, H);
  deriche_kernel2_ct(NDRANGE1(CEILDIV(W, LS)*LS, LS), imgIn, Y2, a3, a4, b1, b2, W, H);
  deriche_kernel3_ct(NDRANGE2(CEILDIV(H, LX), CEILDIV(W, LY), LX, LY), imgOut, Y1, Y2, c1, W, H);
  deriche_kernel4_ct(NDRANGE1(CEILDIV(H, LS)*LS, LS), imgOut, Y1, a5, a6, b1, b2, W, H);
  deriche_kernel5_ct(NDRANGE1(CEILDIV(H, LS)*LS, LS), imgOut, Y2, a7, a8, b1, b2, W, H);
  deriche_kernel3_ct(NDRANGE2(CEILDIV(H, LX), CEILDIV(W, LY), LX, LY), imgOut, Y1, Y2, c2, W, H);
  cc1 = cyc(); REPORT("deriche hwacha-cc", c0, cc1, (long)W*H);
  int bad = check_f("deriche", imgOut, ref, W*H, 1e-4f);
  printf("deriche %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
