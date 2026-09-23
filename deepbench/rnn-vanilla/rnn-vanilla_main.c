// DeepBench vanilla RNN (rnn_bench "vanilla" = cuDNN RNN_RELU, skip-input, one layer), forward pass
// over the training-set shapes (1760 / 2048 / 2560 hidden, batch 16-128, 50 steps) scaled down.
// Kernel: rnn-vanilla.cl (rnn_relu_step, one launch per time step); reference: the same recurrence
// in scalar code, same order.
#include "common.h"
#define MAXH 64
#define MAXB 8
#define MAXT 12
#define LX 8
#define LY 8
void rnn_relu_step_ct(long n, float *x, float *R, float *b, float *h0, float *y, int t, int batch, int hidden);
static float x[MAXT*MAXB*MAXH], R[MAXH*MAXH], b[MAXH], h0[MAXB*MAXH], y[MAXT*MAXB*MAXH], ref[MAXT*MAXB*MAXH];
static struct { int hidden, batch, T; const char *from; } shapes[] = {
  {55, 4, 10, "training 1760x16x50 (/32, /4, /5)"}, {64, 8, 8, "training 2048x128x50 (/32, /16, /6)"}, {40, 2, 12, "training 2560x32x50 (/64, /16, /4)"} };
static int run(int s) {
  int H = shapes[s].hidden, B = shapes[s].batch, T = shapes[s].T;
  for (int i = 0; i < T*B*H; i++) x[i] = frand(-1, 1);
  for (int i = 0; i < H*H; i++) R[i] = frand(-1, 1) / H;
  for (int i = 0; i < H; i++) b[i] = frand(-0.5f, 0.5f);
  for (int i = 0; i < B*H; i++) h0[i] = frand(-1, 1);
  unsigned long c0 = cyc();
  for (int t = 0; t < T; t++) for (int n = 0; n < B; n++) for (int j = 0; j < H; j++) {
    const float *hp = t == 0 ? h0 + n*H : ref + ((t-1)*B + n)*H;
    float acc = x[(t*B + n)*H + j] + b[j]; for (int i = 0; i < H; i++) acc += R[j*H + i] * hp[i];
    ref[(t*B + n)*H + j] = acc > 0 ? acc : 0; }
  unsigned long c1 = cyc(); unsigned long sc = c1 - c0;
  c0 = cyc(); for (int t = 0; t < T; t++) rnn_relu_step_ct(NDRANGE2(CEILDIV(H, LX), CEILDIV(B, LY), LX, LY), x, R, b, h0, y, t, B, H); c1 = cyc();
  printf("rnn-vanilla hidden %d batch %d T %d (%s): scalar %lu cycles, hwacha-cc %lu cycles\n", H, B, T, shapes[s].from, sc, c1 - c0);
  return check_f("  y", y, ref, T*B*H, 1e-4f);
}
int main(void) {
  int bad = 0; for (int s = 0; s < (int)(sizeof shapes / sizeof shapes[0]); s++) bad += run(s);
  printf("rnn-vanilla %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
