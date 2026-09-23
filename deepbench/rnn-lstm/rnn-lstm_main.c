// DeepBench LSTM (rnn_bench "lstm" = cuDNN LSTM, skip-input, one layer), forward pass over the
// training-set shapes (512-4096 hidden, batch 16-128, 25 steps; 256 hidden 150 steps) scaled down.
// Kernel: rnn-lstm.cl (lstm_step, one launch per step); reference: the same recurrence with libm
// expf. hwacha-cc expands exp with a polynomial, so the compare uses a 1e-3 relative tolerance.
#include "common.h"
#define MAXH 64
#define MAXB 8
#define MAXT 12
#define LX 8
#define LY 8
void lstm_step_ct(long n, float *x, float *R, float *b, float *h0, float *c0, float *y, float *cs, int t, int batch, int hidden);
static float x[MAXT*MAXB*MAXH], R[4*MAXH*MAXH], b[4*MAXH], h0[MAXB*MAXH], c0[MAXB*MAXH], y[MAXT*MAXB*MAXH], cs[MAXT*MAXB*MAXH], ref[MAXT*MAXB*MAXH], rc[MAXT*MAXB*MAXH];
static float sigm(float z) { return 1.0f / (1.0f + expf(-z)); }
static float tanh_(float z) { return 2.0f / (1.0f + expf(-2.0f * z)) - 1.0f; }
static struct { int hidden, batch, T; const char *from; } shapes[] = {
  {32, 4, 10, "training 512x16x25 (/16, /4, /2.5)"}, {64, 8, 6, "training 2048x128x25 (/32, /16, /4)"}, {32, 4, 12, "training 256x64x150 (/8, /16, /12)"} };
static int run(int s) {
  int H = shapes[s].hidden, B = shapes[s].batch, T = shapes[s].T;
  for (int i = 0; i < T*B*H; i++) x[i] = frand(-1, 1);
  for (int i = 0; i < 4*H*H; i++) R[i] = frand(-1, 1) / H;
  for (int i = 0; i < 4*H; i++) b[i] = frand(-0.5f, 0.5f);
  for (int i = 0; i < B*H; i++) { h0[i] = frand(-1, 1); c0[i] = frand(-1, 1); }
  unsigned long cy0 = cyc();
  for (int t = 0; t < T; t++) for (int n = 0; n < B; n++) for (int j = 0; j < H; j++) {
    const float *hp = t == 0 ? h0 + n*H : ref + ((t-1)*B + n)*H; float cp = t == 0 ? c0[n*H + j] : rc[((t-1)*B + n)*H + j];
    float xv = x[(t*B + n)*H + j], zi = xv + b[j], zf = xv + b[H + j], zo = xv + b[2*H + j], zg = xv + b[3*H + j];
    for (int i = 0; i < H; i++) { float h = hp[i]; zi += R[j*H + i] * h; zf += R[(H + j)*H + i] * h; zo += R[(2*H + j)*H + i] * h; zg += R[(3*H + j)*H + i] * h; }
    float c = sigm(zf) * cp + sigm(zi) * tanh_(zg); rc[(t*B + n)*H + j] = c; ref[(t*B + n)*H + j] = sigm(zo) * tanh_(c); }
  unsigned long cy1 = cyc(); unsigned long sc = cy1 - cy0;
  cy0 = cyc(); for (int t = 0; t < T; t++) lstm_step_ct(NDRANGE2(CEILDIV(H, LX), CEILDIV(B, LY), LX, LY), x, R, b, h0, c0, y, cs, t, B, H); cy1 = cyc();
  printf("rnn-lstm hidden %d batch %d T %d (%s): scalar %lu cycles, hwacha-cc %lu cycles\n", H, B, T, shapes[s].from, sc, cy1 - cy0);
  int bad = check_f("  h", y, ref, T*B*H, 1e-3f);
  bad += check_f("  c", cs, rc, T*B*H, 1e-3f);
  return bad;
}
int main(void) {
  int bad = 0; for (int s = 0; s < (int)(sizeof shapes / sizeof shapes[0]); s++) bad += run(s);
  printf("rnn-lstm %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
