// Rodinia backprop: one forward step of the input->hidden layer (bpnn_layerforward_ocl: 16x16 work-
// groups, one per 16 input units, products + in-group tree reduction to partial sums) and one weight
// update (bpnn_adjust_weights_ocl), as backprop_ocl.cpp drives them; the reference is a scalar port
// of the same work-group program (the kernel also writes its partially reduced products back)
#include "common.h"
#define IN 64            // input units (multiple of 16)
#define HID 16           // hidden units (= WIDTH)
#define W 16
#define H 16
#define ETA 0.3f
#define MOM 0.3f
#define NB (IN / H)
void bpnn_layerforward_ocl_ct(long n, float *input, float *output_hidden, float *input_hidden, float *partial, float *l_in, float *l_w, int in, int hid);
void bpnn_adjust_weights_ocl_ct(long n, float *delta, int hid, float *ly, int in, float *w, float *oldw);
static float input[IN+1], hidden[HID+1], w[(IN+1)*(HID+1)], rw[(IN+1)*(HID+1)], partial[NB*W], rpartial[NB*W], delta[HID+1], oldw[(IN+1)*(HID+1)], roldw[(IN+1)*(HID+1)], l_in[H], l_w[H*W];
int main(void) {
  for (int i = 0; i <= IN; i++) input[i] = frand(0, 1);
  for (int i = 0; i < (IN+1)*(HID+1); i++) { w[i] = rw[i] = frand(-0.5f, 0.5f); oldw[i] = roldw[i] = frand(-0.1f, 0.1f); }
  for (int j = 0; j <= HID; j++) delta[j] = frand(-0.1f, 0.1f);
  unsigned long c0 = cyc();
  for (int by = 0; by < NB; by++) {                                     // scalar port of one 16x16 work-group
    float wm[H][W], in_node[H];
    for (int ty = 0; ty < H; ty++) in_node[ty] = input[H*by + ty + 1];
    for (int ty = 0; ty < H; ty++) for (int tx = 0; tx < W; tx++) wm[ty][tx] = rw[(HID+1)*H*by + (HID+1)*ty + tx + 1 + (HID+1)] * in_node[ty];
    for (int i = 1; i <= H; i *= 2) { float t[H][W]; memcpy(t, wm, sizeof t);
      for (int ty = 0; ty < H; ty++) if (ty % i == 0) for (int tx = 0; tx < W; tx++) wm[ty][tx] = t[ty][tx] + t[ty + i/2][tx]; }
    for (int ty = 0; ty < H; ty++) for (int tx = 0; tx < W; tx++) rw[(HID+1)*H*by + (HID+1)*ty + tx + 1 + (HID+1)] = wm[ty][tx];
    for (int ty = 0; ty < H; ty++) rpartial[by*HID + ty] = wm[0][ty];
  }
  for (int by = 0; by < NB; by++) for (int ty = 0; ty < H; ty++) for (int tx = 0; tx < W; tx++) {
    int index = (HID+1)*H*by + (HID+1)*ty + tx + 1 + (HID+1), iy = H*by + ty + 1, ix = tx + 1;
    rw[index] += ETA * delta[ix] * input[iy] + MOM * roldw[index];
    roldw[index] = ETA * delta[ix] * input[iy] + MOM * roldw[index];
  }
  for (int tx = 0; tx < W; tx++) { int ix = tx + 1; rw[ix] += ETA * delta[ix] + MOM * roldw[ix]; roldw[ix] = ETA * delta[ix] + MOM * roldw[ix]; }
  unsigned long c1 = cyc(); REPORT("backprop scalar", c0, c1, (long)IN*HID);
  c0 = cyc();
  bpnn_layerforward_ocl_ct(NDRANGE2(1, NB, W, H), input, hidden, w, partial, l_in, l_w, IN, HID);
  bpnn_adjust_weights_ocl_ct(NDRANGE2(1, NB, W, H), delta, HID, input, IN, w, oldw);
  c1 = cyc(); REPORT("backprop hwacha-cc", c0, c1, (long)IN*HID);
  int bad = check_f("partial_sum", partial, rpartial, NB*W, 1e-4f) + check_f("w", w, rw, (IN+1)*(HID+1), 1e-4f) + check_f("oldw", oldw, roldw, (IN+1)*(HID+1), 1e-4f);
  printf("backprop %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
