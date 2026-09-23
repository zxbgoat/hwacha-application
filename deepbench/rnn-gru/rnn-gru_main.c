// DeepBench GRU (rnn_bench "gru" = cuDNN GRU, skip-input, one layer), forward pass over the
// training-set shapes (2816 / 2048 / 1536 / 2560 hidden, batch 32, 187-1500 steps; 1024 hidden batch
// 64) scaled down. Kernel: rnn-gru.cl (gru_step, one launch per step); reference: the same
// recurrence with libm expf (1e-3 relative tolerance for hwacha-cc's exp expansion).
#include "common.h"
#define MAXH 64
#define MAXB 8
#define MAXT 12
#define LX 8
#define LY 8
void gru_step_ct(long n, float *x, float *R, float *bW, float *bR, float *h0, float *y, int t, int batch, int hidden);
static float x[MAXT*MAXB*MAXH], R[3*MAXH*MAXH], bW[3*MAXH], bR[3*MAXH], h0[MAXB*MAXH], y[MAXT*MAXB*MAXH], ref[MAXT*MAXB*MAXH];
static float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }
static float tanh_(float v) { return 2.0f / (1.0f + expf(-2.0f * v)) - 1.0f; }
static struct { int hidden, batch, T; const char *from; } shapes[] = {
  {44, 4, 12, "training 2816x32x1500 (/64, /8, /125)"}, {64, 4, 6, "training 2048x32x187 (/32, /8, /31)"}, {32, 8, 10, "training 1024x64x1500 (/32, /8, /150)"} };
static int run(int s) {
  int H = shapes[s].hidden, B = shapes[s].batch, T = shapes[s].T;
  for (int i = 0; i < T*B*H; i++) x[i] = frand(-1, 1);
  for (int i = 0; i < 3*H*H; i++) R[i] = frand(-1, 1) / H;
  for (int i = 0; i < 3*H; i++) { bW[i] = frand(-0.5f, 0.5f); bR[i] = frand(-0.5f, 0.5f); }
  for (int i = 0; i < B*H; i++) h0[i] = frand(-1, 1);
  unsigned long cy0 = cyc();
  for (int t = 0; t < T; t++) for (int n = 0; n < B; n++) for (int j = 0; j < H; j++) {
    const float *hp = t == 0 ? h0 + n*H : ref + ((t-1)*B + n)*H;
    float xv = x[(t*B + n)*H + j], zr = xv + bW[j] + bR[j], zz = xv + bW[H + j] + bR[H + j], zh = bR[2*H + j];
    for (int i = 0; i < H; i++) { float h = hp[i]; zr += R[j*H + i] * h; zz += R[(H + j)*H + i] * h; zh += R[(2*H + j)*H + i] * h; }
    float r = sigm(zr), z = sigm(zz), hn = tanh_(xv + r * zh + bW[2*H + j]);
    ref[(t*B + n)*H + j] = (1.0f - z) * hn + z * hp[j]; }
  unsigned long cy1 = cyc(); unsigned long sc = cy1 - cy0;
  cy0 = cyc(); for (int t = 0; t < T; t++) gru_step_ct(NDRANGE2(CEILDIV(H, LX), CEILDIV(B, LY), LX, LY), x, R, bW, bR, h0, y, t, B, H); cy1 = cyc();
  printf("rnn-gru hidden %d batch %d T %d (%s): scalar %lu cycles, hwacha-cc %lu cycles\n", H, B, T, shapes[s].from, sc, cy1 - cy0);
  return check_f("  h", y, ref, T*B*H, 1e-3f);
}
int main(void) {
  int bad = 0; for (int s = 0; s < (int)(sizeof shapes / sizeof shapes[0]); s++) bad += run(s);
  printf("rnn-gru %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
