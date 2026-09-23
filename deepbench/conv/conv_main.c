// DeepBench convolution (conv_bench: forward, backward data and backward filter, NCHW, cuDNN
// cross-correlation semantics), on shapes of the training set scaled down: the 3x3 / pad 1 /
// stride 1 layers (VGG / ResNet), the 1x1 stride-2 projection, the 5x5 / pad 2 inception layer and
// the DeepSpeech 20x5 / stride 2 unpadded layer (as 5x5 here). Kernels: conv.cl; reference: the same
// loops in scalar code (same summation order, so the compare is exact).
#include "common.h"
#define MAXX (4*16*16*16)
#define MAXF (16*16*5*5)
#define LX 8
#define LY 8
typedef void conv_fn(long n, float *a, float *b, float *c, int N, int C, int H, int W, int K, int R, int S, int pad_h, int pad_w, int hs, int ws, int OH, int OW);
conv_fn conv_fwd_ct, conv_bwd_data_ct, conv_bwd_filter_ct;
static float x[MAXX], f[MAXF], y[MAXX], dy[MAXX], dx[MAXX], df[MAXF], ry[MAXX], rdx[MAXX], rdf[MAXF];
static struct { int w, h, c, n, k, s, r, pad_w, pad_h, ws, hs; const char *from; } shapes[] = {
  {14, 14, 16, 2, 16, 3, 3, 1, 1, 1, 1, "VGG 14x14x512 n16 k512 3x3 (c,k /32, n /8)"},
  {16, 16, 8, 2, 16, 1, 1, 0, 0, 2, 2, "ResNet 56x56x64 n8 k256 1x1 stride 2 (/3.5, c /8, k /16)"},
  {14, 14, 12, 2, 8, 5, 5, 2, 2, 1, 1, "inception 28x28x192 n16 k32 5x5 pad 2 (/2, c /16, k /4)"},
  {16, 16, 1, 2, 8, 5, 5, 0, 0, 2, 2, "DeepSpeech 700x161x1 n4 k32 20x5 stride 2 (5x5, /40, k /4)"},
  {12, 8, 4, 3, 12, 3, 3, 1, 1, 1, 1, "DeepSpeech 120x12x32 n16 k64 3x3 (/10, c /8, k /5)"} };
static int run(int s) {
  int W = shapes[s].w, H = shapes[s].h, C = shapes[s].c, N = shapes[s].n, K = shapes[s].k, S = shapes[s].s, R = shapes[s].r, pw = shapes[s].pad_w, ph = shapes[s].pad_h, ws = shapes[s].ws, hs = shapes[s].hs;
  int OH = (H + 2*ph - R) / hs + 1, OW = (W + 2*pw - S) / ws + 1;
  for (int i = 0; i < N*C*H*W; i++) x[i] = frand(-1, 1);
  for (int i = 0; i < K*C*R*S; i++) f[i] = frand(-1, 1);
  for (int i = 0; i < N*K*OH*OW; i++) dy[i] = frand(-1, 1);
  unsigned long c0 = cyc();
  for (int n = 0; n < N; n++) for (int k = 0; k < K; k++) for (int oy = 0; oy < OH; oy++) for (int ox = 0; ox < OW; ox++) {
    float acc = 0;
    for (int c = 0; c < C; c++) for (int fy = 0; fy < R; fy++) { int iy = oy*hs - ph + fy; if (iy < 0 || iy >= H) continue; for (int fx = 0; fx < S; fx++) { int ix = ox*ws - pw + fx; if (ix < 0 || ix >= W) continue; acc += x[((n*C + c)*H + iy)*W + ix] * f[((k*C + c)*R + fy)*S + fx]; } }
    ry[((n*K + k)*OH + oy)*OW + ox] = acc; }
  for (int n = 0; n < N; n++) for (int c = 0; c < C; c++) for (int iy = 0; iy < H; iy++) for (int ix = 0; ix < W; ix++) {
    float acc = 0;
    for (int k = 0; k < K; k++) for (int fy = 0; fy < R; fy++) { int ty = iy + ph - fy; if (ty < 0 || ty % hs != 0) continue; int oy = ty / hs; if (oy >= OH) continue; for (int fx = 0; fx < S; fx++) { int tx = ix + pw - fx; if (tx < 0 || tx % ws != 0) continue; int ox = tx / ws; if (ox >= OW) continue; acc += dy[((n*K + k)*OH + oy)*OW + ox] * f[((k*C + c)*R + fy)*S + fx]; } }
    rdx[((n*C + c)*H + iy)*W + ix] = acc; }
  for (int k = 0; k < K; k++) for (int c = 0; c < C; c++) for (int fy = 0; fy < R; fy++) for (int fx = 0; fx < S; fx++) {
    float acc = 0;
    for (int n = 0; n < N; n++) for (int oy = 0; oy < OH; oy++) { int iy = oy*hs - ph + fy; if (iy < 0 || iy >= H) continue; for (int ox = 0; ox < OW; ox++) { int ix = ox*ws - pw + fx; if (ix < 0 || ix >= W) continue; acc += dy[((n*K + k)*OH + oy)*OW + ox] * x[((n*C + c)*H + iy)*W + ix]; } }
    rdf[((k*C + c)*R + fy)*S + fx] = acc; }
  unsigned long c1 = cyc(); unsigned long sc = c1 - c0;
  c0 = cyc();
  conv_fwd_ct(NDRANGE2(CEILDIV(OH*OW, LX), CEILDIV(N*K, LY), LX, LY), x, f, y, N, C, H, W, K, R, S, ph, pw, hs, ws, OH, OW);
  conv_bwd_data_ct(NDRANGE2(CEILDIV(H*W, LX), CEILDIV(N*C, LY), LX, LY), dy, f, dx, N, C, H, W, K, R, S, ph, pw, hs, ws, OH, OW);
  conv_bwd_filter_ct(NDRANGE2(CEILDIV(R*S, LX), CEILDIV(K*C, LY), LX, LY), dy, x, df, N, C, H, W, K, R, S, ph, pw, hs, ws, OH, OW);
  c1 = cyc();
  printf("conv %dx%dx%d n%d k%d %dx%d pad %d,%d stride %d,%d (%s): scalar %lu cycles, hwacha-cc %lu cycles\n", W, H, C, N, K, S, R, pw, ph, ws, hs, shapes[s].from, sc, c1 - c0);
  int bad = check_f("  fwd", y, ry, N*K*OH*OW, 1e-4f);
  bad += check_f("  bwd_data", dx, rdx, N*C*H*W, 1e-4f);
  bad += check_f("  bwd_filter", df, rdf, K*C*R*S, 1e-4f);
  return bad;
}
int main(void) {
  int bad = 0; for (int s = 0; s < (int)(sizeof shapes / sizeof shapes[0]); s++) bad += run(s);
  printf("conv %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
