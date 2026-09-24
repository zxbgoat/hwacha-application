// DeepBench LSTM in half precision (rnn_bench "lstm" with "half": half x / R / b / h / c, float
// gate math), forward pass over the training-set shapes scaled down. Kernel: rnn-lstm-fp16.cl
// (lstm_step, one launch per step); reference: the same recurrence in float with libm expf, the
// state rounded to half after every step as the kernel stores it. hwacha-cc expands exp with a
// polynomial and the state is quantised to half, so the compare uses a 4e-3 relative tolerance.
#include "common.h"
#define MAXH 64
#define MAXB 8
#define MAXT 12
#define LX 8
#define LY 8
void lstm_step_ct(long n, f16 *x, f16 *R, f16 *b, f16 *h0, f16 *c0, f16 *y, f16 *cs, int t, int batch, int hidden);
static f16 xh[MAXT*MAXB*MAXH], Rh[4*MAXH*MAXH], bh[4*MAXH], h0h[MAXB*MAXH], c0h[MAXB*MAXH], y[MAXT*MAXB*MAXH], cs[MAXT*MAXB*MAXH];
static float x[MAXT*MAXB*MAXH], R[4*MAXH*MAXH], b[4*MAXH], h0[MAXB*MAXH], c0[MAXB*MAXH], yf[MAXT*MAXB*MAXH], csf[MAXT*MAXB*MAXH], ref[MAXT*MAXB*MAXH], rc[MAXT*MAXB*MAXH];
static float sigm(float z) { return 1.0f / (1.0f + expf(-z)); }
static float tanh_(float z) { return 2.0f / (1.0f + expf(-2.0f * z)) - 1.0f; }
static struct { int hidden, batch, T; const char *from; } shapes[] = {
  {32, 4, 10, "training 512x16x25 (/16, /4, /2.5)"}, {64, 8, 6, "training 2048x128x25 (/32, /16, /4)"}, {32, 4, 12, "training 256x64x150 (/8, /16, /12)"} };
static int run(int s) {
  int H = shapes[s].hidden, B = shapes[s].batch, T = shapes[s].T;
  for (int i = 0; i < T*B*H; i++) { xh[i] = f32_to_f16(frand(-1, 1)); x[i] = f16_to_f32(xh[i]); }
  for (int i = 0; i < 4*H*H; i++) { Rh[i] = f32_to_f16(frand(-1, 1) / H); R[i] = f16_to_f32(Rh[i]); }
  for (int i = 0; i < 4*H; i++) { bh[i] = f32_to_f16(frand(-0.5f, 0.5f)); b[i] = f16_to_f32(bh[i]); }
  for (int i = 0; i < B*H; i++) { h0h[i] = f32_to_f16(frand(-1, 1)); h0[i] = f16_to_f32(h0h[i]); c0h[i] = f32_to_f16(frand(-1, 1)); c0[i] = f16_to_f32(c0h[i]); }
  unsigned long cy0 = cyc();
  for (int t = 0; t < T; t++) for (int n = 0; n < B; n++) for (int j = 0; j < H; j++) {
    const float *hp = t == 0 ? h0 + n*H : ref + ((t-1)*B + n)*H; float cp = t == 0 ? c0[n*H + j] : rc[((t-1)*B + n)*H + j];
    float xv = x[(t*B + n)*H + j], zi = xv + b[j], zf = xv + b[H + j], zo = xv + b[2*H + j], zg = xv + b[3*H + j];
    for (int i = 0; i < H; i++) { float h = hp[i]; zi += R[j*H + i] * h; zf += R[(H + j)*H + i] * h; zo += R[(2*H + j)*H + i] * h; zg += R[(3*H + j)*H + i] * h; }
    float c = sigm(zf) * cp + sigm(zi) * tanh_(zg); rc[(t*B + n)*H + j] = h_round(c); ref[(t*B + n)*H + j] = h_round(sigm(zo) * tanh_(c)); }
  unsigned long cy1 = cyc(); unsigned long sc = cy1 - cy0;
  cy0 = cyc(); for (int t = 0; t < T; t++) lstm_step_ct(NDRANGE2(CEILDIV(H, LX), CEILDIV(B, LY), LX, LY), xh, Rh, bh, h0h, c0h, y, cs, t, B, H); cy1 = cyc();
  for (int i = 0; i < T*B*H; i++) { yf[i] = f16_to_f32(y[i]); csf[i] = f16_to_f32(cs[i]); }
  printf("rnn-lstm-fp16 hidden %d batch %d T %d (%s): scalar %lu cycles, hwacha-cc %lu cycles\n", H, B, T, shapes[s].from, sc, cy1 - cy0);
  int bad = check_f("  h", yf, ref, T*B*H, 4e-3f);
  bad += check_f("  c", csf, rc, T*B*H, 4e-3f);
  return bad;
}
int main(void) {
  int bad = 0; for (int s = 0; s < (int)(sizeof shapes / sizeof shapes[0]); s++) bad += run(s);
  printf("rnn-lstm-fp16 %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
