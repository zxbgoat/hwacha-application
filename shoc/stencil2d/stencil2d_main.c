// SHOC Stencil2D (level1): nIters iterations of the 9-point stencil (centre 0.25, cardinal 0.15,
// diagonal 0.05, SHOC's defaults) on a matrix with a one-element halo of zeros, with the SHOC
// StencilKernel (each work-item computes LROWS rows of one column from a __local haloed tile) and
// CopyRect (carries the left / right halo columns to the new buffer), launched as OpenCLStencil.cpp
// does: work-groups of 1 x LCOLS work-items, (rows-2)/LROWS x (cols-2)/LCOLS groups, the buffers
// swapped every iteration. Rows are padded to a multiple of 16 columns (Matrix2D's pad, the
// `alignment` argument). Interior random in [0, 1); validation: relative error below 0.01 (SHOC's
// val-threshold). Size 66 x 66 (64 x 64 interior), 10 iterations (SHOC: 1000).
#include "common.h"
#define ROWS 66
#define COLS 66
#define PAD 16
#define PCOLS (COLS + (COLS % PAD ? PAD - COLS % PAD : 0))
#define LROWS 8
#define LCOLS 8
#define NITERS 10
void CopyRect_ct(long n, float *dest, int doffset, int dpitch, float *src, int soffset, int spitch, int width, int height);
void StencilKernel_ct(long n, float *data, float *newData, int alignment, float wCenter, float wCardinal, float wDiagonal, float *sh);
static float A[ROWS*PCOLS], B[ROWS*PCOLS], ref[ROWS*PCOLS], tmp[ROWS*PCOLS], sh[(LROWS+2)*(LCOLS+2)];
int main(void) {
  const float wc = 0.25f, wca = 0.15f, wd = 0.05f;
  for (int r = 0; r < ROWS; r++) for (int c = 0; c < PCOLS; c++) { float v = (r >= 1 && r < ROWS-1 && c >= 1 && c < COLS-1) ? frand(0, 1) : 0; A[r*PCOLS + c] = B[r*PCOLS + c] = ref[r*PCOLS + c] = v; }
  unsigned long c0 = cyc();
  for (int it = 0; it < NITERS; it++) {
    for (int r = 1; r < ROWS-1; r++) for (int c = 1; c < COLS-1; c++) {
      const float *p = ref + r*PCOLS + c;
      tmp[r*PCOLS + c] = wc * p[0] + wca * (p[-PCOLS] + p[PCOLS] + p[1] + p[-1]) + wd * (p[-PCOLS+1] + p[PCOLS+1] + p[-PCOLS-1] + p[PCOLS-1]); }
    for (int r = 1; r < ROWS-1; r++) for (int c = 1; c < COLS-1; c++) ref[r*PCOLS + c] = tmp[r*PCOLS + c];
  }
  unsigned long c1 = cyc(); REPORT("stencil2d scalar", c0, c1, (long)(ROWS-2)*(COLS-2)*NITERS);
  float *cur = A, *nw = B;
  c0 = cyc();
  for (int it = 0; it < NITERS; it++) {
    CopyRect_ct(NDRANGE1(CEILDIV(ROWS, LCOLS)*LCOLS, LCOLS), nw, 0, PCOLS, cur, 0, PCOLS, 1, ROWS);
    CopyRect_ct(NDRANGE1(CEILDIV(ROWS, LCOLS)*LCOLS, LCOLS), nw, COLS-1, PCOLS, cur, COLS-1, PCOLS, 1, ROWS);
    StencilKernel_ct(NDRANGE2((ROWS-2)/LROWS, (COLS-2)/LCOLS, 1, LCOLS), cur, nw, PAD, wc, wca, wd, sh);
    float *t = cur; cur = nw; nw = t;
  }
  c1 = cyc(); REPORT("stencil2d hwacha-cc", c0, c1, (long)(ROWS-2)*(COLS-2)*NITERS);
  int bad = 0; float md = 0;
  for (int r = 1; r < ROWS-1; r++) for (int c = 1; c < COLS-1; c++) { float e = fabsf(cur[r*PCOLS + c] - ref[r*PCOLS + c]) / (fabsf(ref[r*PCOLS + c]) + 1e-6f); if (e > md) md = e; if (e > 0.01f) bad++; }
  printf("stencil2d: %d interior values above the 0.01 threshold / %d, max relative error %ld/1e6\n", bad, (ROWS-2)*(COLS-2), (long)(md*1e6f));
  printf("stencil2d %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
