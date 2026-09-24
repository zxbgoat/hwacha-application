// DeepBench vanilla RNN in half precision (rnn_bench "vanilla" with "half": half x / R / b / h, float
// math), forward pass over the training-set shapes scaled down. Kernel: rnn-vanilla-fp16.cl
// (rnn_relu_step, one launch per time step); reference: the same recurrence in float on the half
// inputs, the state rounded to half every step as the kernel stores it (exact compare).
#include "common.h"
#define MAXH 64
#define MAXB 8
#define MAXT 12
#define LX 8
#define LY 8
void rnn_relu_step_ct(long n, f16 *x, f16 *R, f16 *b, f16 *h0, f16 *y, int t, int batch, int hidden);
static f16 xh[MAXT*MAXB*MAXH], Rh[MAXH*MAXH], bh[MAXH], h0h[MAXB*MAXH], y[MAXT*MAXB*MAXH];
static float x[MAXT*MAXB*MAXH], R[MAXH*MAXH], b[MAXH], h0[MAXB*MAXH], yf[MAXT*MAXB*MAXH], ref[MAXT*MAXB*MAXH];
static struct { int hidden, batch, T; const char *from; } shapes[] = {
  {55, 4, 10, "training 1760x16x50 (/32, /4, /5)"}, {64, 8, 8, "training 2048x128x50 (/32, /16, /6)"}, {40, 2, 12, "training 2560x32x50 (/64, /16, /4)"} };
static int run(int s) {
  int H = shapes[s].hidden, B = shapes[s].batch, T = shapes[s].T;
  for (int i = 0; i < T*B*H; i++) { xh[i] = f32_to_f16(frand(-1, 1)); x[i] = f16_to_f32(xh[i]); }
  for (int i = 0; i < H*H; i++) { Rh[i] = f32_to_f16(frand(-1, 1) / H); R[i] = f16_to_f32(Rh[i]); }
  for (int i = 0; i < H; i++) { bh[i] = f32_to_f16(frand(-0.5f, 0.5f)); b[i] = f16_to_f32(bh[i]); }
  for (int i = 0; i < B*H; i++) { h0h[i] = f32_to_f16(frand(-1, 1)); h0[i] = f16_to_f32(h0h[i]); }
  unsigned long c0 = cyc();
  for (int t = 0; t < T; t++) for (int n = 0; n < B; n++) for (int j = 0; j < H; j++) {
    const float *hp = t == 0 ? h0 + n*H : ref + ((t-1)*B + n)*H;
    float acc = x[(t*B + n)*H + j] + b[j]; for (int i = 0; i < H; i++) acc += R[j*H + i] * hp[i];
    ref[(t*B + n)*H + j] = h_round(acc > 0 ? acc : 0); }
  unsigned long c1 = cyc(); unsigned long sc = c1 - c0;
  c0 = cyc(); for (int t = 0; t < T; t++) rnn_relu_step_ct(NDRANGE2(CEILDIV(H, LX), CEILDIV(B, LY), LX, LY), xh, Rh, bh, h0h, y, t, B, H); c1 = cyc();
  for (int i = 0; i < T*B*H; i++) yf[i] = f16_to_f32(y[i]);
  printf("rnn-vanilla-fp16 hidden %d batch %d T %d (%s): scalar %lu cycles, hwacha-cc %lu cycles\n", H, B, T, shapes[s].from, sc, c1 - c0);
  return check_f("  y", yf, ref, T*B*H, 1e-5f);
}
int main(void) {
  int bad = 0; for (int s = 0; s < (int)(sizeof shapes / sizeof shapes[0]); s++) bad += run(s);
  printf("rnn-vanilla-fp16 %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
