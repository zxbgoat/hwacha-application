// DeepBench GRU in half precision (rnn_bench "gru" with "half": half x / R / biases / h, float gate
// math), forward pass over the training-set shapes scaled down. Kernel: rnn-gru-fp16.cl (gru_step,
// one launch per step); reference: the same recurrence in float with libm expf, the state rounded
// to half every step as the kernel stores it (4e-3 relative tolerance: exp expansion + half state).
#include "common.h"
#define MAXH 64
#define MAXB 8
#define MAXT 12
#define LX 8
#define LY 8
void gru_step_ct(long n, f16 *x, f16 *R, f16 *bW, f16 *bR, f16 *h0, f16 *y, int t, int batch, int hidden);
static f16 xh[MAXT*MAXB*MAXH], Rh[3*MAXH*MAXH], bWh[3*MAXH], bRh[3*MAXH], h0h[MAXB*MAXH], y[MAXT*MAXB*MAXH];
static float x[MAXT*MAXB*MAXH], R[3*MAXH*MAXH], bW[3*MAXH], bR[3*MAXH], h0[MAXB*MAXH], yf[MAXT*MAXB*MAXH], ref[MAXT*MAXB*MAXH];
static float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }
static float tanh_(float v) { return 2.0f / (1.0f + expf(-2.0f * v)) - 1.0f; }
static struct { int hidden, batch, T; const char *from; } shapes[] = {
  {44, 4, 12, "training 2816x32x1500 (/64, /8, /125)"}, {64, 4, 6, "training 2048x32x187 (/32, /8, /31)"}, {32, 8, 10, "training 1024x64x1500 (/32, /8, /150)"} };
static int run(int s) {
  int H = shapes[s].hidden, B = shapes[s].batch, T = shapes[s].T;
  for (int i = 0; i < T*B*H; i++) { xh[i] = f32_to_f16(frand(-1, 1)); x[i] = f16_to_f32(xh[i]); }
  for (int i = 0; i < 3*H*H; i++) { Rh[i] = f32_to_f16(frand(-1, 1) / H); R[i] = f16_to_f32(Rh[i]); }
  for (int i = 0; i < 3*H; i++) { bWh[i] = f32_to_f16(frand(-0.5f, 0.5f)); bW[i] = f16_to_f32(bWh[i]); bRh[i] = f32_to_f16(frand(-0.5f, 0.5f)); bR[i] = f16_to_f32(bRh[i]); }
  for (int i = 0; i < B*H; i++) { h0h[i] = f32_to_f16(frand(-1, 1)); h0[i] = f16_to_f32(h0h[i]); }
  unsigned long cy0 = cyc();
  for (int t = 0; t < T; t++) for (int n = 0; n < B; n++) for (int j = 0; j < H; j++) {
    const float *hp = t == 0 ? h0 + n*H : ref + ((t-1)*B + n)*H;
    float xv = x[(t*B + n)*H + j], zr = xv + bW[j] + bR[j], zz = xv + bW[H + j] + bR[H + j], zh = bR[2*H + j];
    for (int i = 0; i < H; i++) { float h = hp[i]; zr += R[j*H + i] * h; zz += R[(H + j)*H + i] * h; zh += R[(2*H + j)*H + i] * h; }
    float r = sigm(zr), z = sigm(zz), hn = tanh_(xv + r * zh + bW[2*H + j]);
    ref[(t*B + n)*H + j] = h_round((1.0f - z) * hn + z * hp[j]); }
  unsigned long cy1 = cyc(); unsigned long sc = cy1 - cy0;
  cy0 = cyc(); for (int t = 0; t < T; t++) gru_step_ct(NDRANGE2(CEILDIV(H, LX), CEILDIV(B, LY), LX, LY), xh, Rh, bWh, bRh, h0h, y, t, B, H); cy1 = cyc();
  for (int i = 0; i < T*B*H; i++) yf[i] = f16_to_f32(y[i]);
  printf("rnn-gru-fp16 hidden %d batch %d T %d (%s): scalar %lu cycles, hwacha-cc %lu cycles\n", H, B, T, shapes[s].from, sc, cy1 - cy0);
  return check_f("  h", yf, ref, T*B*H, 4e-3f);
}
int main(void) {
  int bad = 0; for (int s = 0; s < (int)(sizeof shapes / sizeof shapes[0]); s++) bad += run(s);
  printf("rnn-gru-fp16 %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
