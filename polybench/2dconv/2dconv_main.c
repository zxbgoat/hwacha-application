// PolyBench 2DConvolution (2dconv): 3x3 stencil over an NI x NJ image, one work-item per pixel.
// The kernel is the unmodified PolyBench/GPU Convolution2D_kernel.cl (multiplication by 0.2 stops
// clang from contracting the stencil into an FMA, which would not match the reference exactly).
#include "common.h"
#define NI 64
#define NJ 64
#define LX 8   // PolyBench/GPU uses 32 x 8 = 256 lanes; 8 x 8 fits a Hwacha group
#define LY 8
void Convolution2D_kernel_ct(long n, float *A, float *B, int ni, int nj);
static float A[NI*NJ], B[NI*NJ], ref[NI*NJ];
static void ref_kernel(const float *A, float *B, int ni, int nj) {
  for (int i = 1; i < ni - 1; i++) for (int j = 1; j < nj - 1; j++)
    B[i*nj + j] = 0.2f*A[(i-1)*nj + (j-1)] + 0.5f*A[(i-1)*nj + j] + -0.8f*A[(i-1)*nj + (j+1)]
                + -0.3f*A[i*nj + (j-1)]   + 0.6f*A[i*nj + j]     + -0.9f*A[i*nj + (j+1)]
                + 0.4f*A[(i+1)*nj + (j-1)] + 0.7f*A[(i+1)*nj + j] + 0.10f*A[(i+1)*nj + (j+1)];
}
int main(void) {
  for (int i = 0; i < NI; i++) for (int j = 0; j < NJ; j++) A[i*NJ + j] = frand(0, 1);
  unsigned long c0 = cyc(); ref_kernel(A, ref, NI, NJ); unsigned long c1 = cyc();
  REPORT("2dconv scalar", c0, c1, (long)NI*NJ);
  c0 = cyc(); Convolution2D_kernel_ct(NDRANGE2(CEILDIV(NJ, LX), CEILDIV(NI, LY), LX, LY), A, B, NI, NJ); c1 = cyc();
  REPORT("2dconv hwacha-cc", c0, c1, (long)NI*NJ);
  int bad = check_f("2dconv", B, ref, NI*NJ, 1e-3f);
  printf("2dconv %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
