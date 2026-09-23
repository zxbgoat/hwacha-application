// PolyBench 3DConvolution (3dconv): the PolyBench/GPU 3-D stencil, one 2-D launch (j, k) per plane
// i, as the original host does. Kernel: the unmodified Convolution3D_kernel; reference: conv3D of
// that host (same stencil, integer coefficients on integer-valued inputs, so the compare is exact).
#include "common.h"
#define NI 16
#define NJ 32
#define NK 32
#define LX 8
#define LY 8
void Convolution3D_kernel_ct(long n, float *A, float *B, int ni, int nj, int nk, int i);
static float A[NI*NJ*NK], B[NI*NJ*NK], ref[NI*NJ*NK];
#define IDX(i, j, k) ((i)*(NK*NJ) + (j)*NK + (k))
static void ref_kernel(void) {
  const float c11 = 2, c21 = 5, c31 = -8, c12 = -3, c22 = 6, c32 = -9, c13 = 4, c23 = 7, c33 = 10;
  for (int i = 1; i < NI - 1; i++) for (int j = 1; j < NJ - 1; j++) for (int k = 1; k < NK - 1; k++)
    ref[IDX(i,j,k)] = c11 * A[IDX(i-1,j-1,k-1)] + c13 * A[IDX(i+1,j-1,k-1)]
                    + c21 * A[IDX(i-1,j-1,k-1)] + c23 * A[IDX(i+1,j-1,k-1)]
                    + c31 * A[IDX(i-1,j-1,k-1)] + c33 * A[IDX(i+1,j-1,k-1)]
                    + c12 * A[IDX(i,j-1,k)]     + c22 * A[IDX(i,j,k)]
                    + c32 * A[IDX(i,j+1,k)]     + c11 * A[IDX(i-1,j-1,k+1)]
                    + c13 * A[IDX(i+1,j-1,k+1)] + c21 * A[IDX(i-1,j,k+1)]
                    + c23 * A[IDX(i+1,j,k+1)]   + c31 * A[IDX(i-1,j+1,k+1)]
                    + c33 * A[IDX(i+1,j+1,k+1)];
}
int main(void) {
  for (int i = 0; i < NI; i++) for (int j = 0; j < NJ; j++) for (int k = 0; k < NK; k++) A[IDX(i,j,k)] = i % 12 + 2 * (j % 7) + 3 * (k % 13);
  unsigned long c0 = cyc(); ref_kernel(); unsigned long c1 = cyc();
  REPORT("3dconv scalar", c0, c1, (long)NI*NJ*NK);
  c0 = cyc();
  for (int i = 1; i < NI - 1; i++) Convolution3D_kernel_ct(NDRANGE2(CEILDIV(NK, LX), CEILDIV(NJ, LY), LX, LY), A, B, NI, NJ, NK, i);
  c1 = cyc(); REPORT("3dconv hwacha-cc", c0, c1, (long)NI*NJ*NK);
  int bad = check_f("3dconv", B, ref, NI*NJ*NK, 1e-4f);   // the kernel writes 0 outside the interior, ref is 0 there too
  printf("3dconv %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
