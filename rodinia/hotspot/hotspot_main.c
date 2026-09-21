// Rodinia hotspot: 2-D thermal stencil in BLOCK_SIZE x BLOCK_SIZE tiles with a pyramid of `iteration`
// steps per launch (hotspot.c's compute_tran_temp), against a scalar stencil with clamped neighbours
#include "common.h"
#define BS 8
#define ROWS 64
#define COLS 64
#define PYR 2          // pyramid_height: steps per launch
#define TOTAL 4        // total_iterations
#define EXPAND_RATE 2
void hotspot_ct(long n, int iteration, float *power, float *temp_src, float *temp_dst, int grid_cols, int grid_rows,
                int border_cols, int border_rows, float Cap, float Rx, float Ry, float Rz, float step);
static float power[ROWS*COLS], t0[ROWS*COLS], t1[ROWS*COLS], r0[ROWS*COLS], r1[ROWS*COLS];
int main(void) {
  for (int i = 0; i < ROWS*COLS; i++) { power[i] = frand(0, 1e-3f); t0[i] = r0[i] = frand(320, 340); }
  // chip parameters as hotspot.c (grid 64x64 of a 16mm chip)
  float t_chip = 0.0005f, chip_height = 0.016f, chip_width = 0.016f, K_SI = 100, FACTOR_CHIP = 0.5f, SPEC_HEAT_SI = 1.75e6f, MAX_PD = 3e6f, PRECISION = 0.001f;
  float grid_height = chip_height / ROWS, grid_width = chip_width / COLS;
  float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
  float Rx = grid_width / (2.0f * K_SI * t_chip * grid_height), Ry = grid_height / (2.0f * K_SI * t_chip * grid_width), Rz = t_chip / (K_SI * grid_height * grid_width);
  float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI), step = PRECISION / max_slope;
  int borderCols = PYR * EXPAND_RATE / 2, borderRows = PYR * EXPAND_RATE / 2;
  int smallBlock = BS - PYR * EXPAND_RATE, blockCols = CEILDIV(COLS, smallBlock), blockRows = CEILDIV(ROWS, smallBlock);
  // scalar reference: TOTAL plain stencil steps (the tiles' clamped N/S/W/E equal a replicated border)
  unsigned long c0 = cyc();
  float step_div_Cap = step / Cap, Rx_1 = 1 / Rx, Ry_1 = 1 / Ry, Rz_1 = 1 / Rz, amb = 80.0f;
  float *src = r0, *dst = r1;
  for (int it = 0; it < TOTAL; it++) {
    for (int y = 0; y < ROWS; y++) for (int x = 0; x < COLS; x++) {
      int N = y > 0 ? y - 1 : 0, S = y < ROWS - 1 ? y + 1 : ROWS - 1, W = x > 0 ? x - 1 : 0, E = x < COLS - 1 ? x + 1 : COLS - 1;
      float t = src[y*COLS+x];
      dst[y*COLS+x] = t + step_div_Cap * (power[y*COLS+x] + (src[S*COLS+x] + src[N*COLS+x] - 2.0f * t) * Ry_1 + (src[y*COLS+E] + src[y*COLS+W] - 2.0f * t) * Rx_1 + (amb - t) * Rz_1);
    }
    float *tmp = src; src = dst; dst = tmp;
  }
  unsigned long c1 = cyc(); REPORT("hotspot scalar", c0, c1, (long)ROWS*COLS*TOTAL);
  float *ref = src;
  c0 = cyc();
  float *hsrc = t0, *hdst = t1;
  for (int t = 0; t < TOTAL; t += PYR) {
    int iter = TOTAL - t < PYR ? TOTAL - t : PYR;
    hotspot_ct(NDRANGE2(blockCols, blockRows, BS, BS), iter, power, hsrc, hdst, COLS, ROWS, borderCols, borderRows, Cap, Rx, Ry, Rz, step);
    float *tmp = hsrc; hsrc = hdst; hdst = tmp;
  }
  c1 = cyc(); REPORT("hotspot hwacha-cc", c0, c1, (long)ROWS*COLS*TOTAL);
  int bad = check_f("temp", hsrc, ref, ROWS*COLS, 1e-4f);
  printf("hotspot %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
