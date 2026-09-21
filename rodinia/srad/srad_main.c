// Rodinia srad: speckle-reducing anisotropic diffusion on an Nr x Nc image: extract (exp), then per
// iteration prepare + tree reduce (mean / variance of the image -> q0sqr), srad_kernel (diffusion
// coefficients) and srad2_kernel (update), then compress (log), as kernel_gpu_opencl_wrapper.c drives
// them; against a scalar copy with a plain-loop reduction
#include "common.h"
#define NR 32
#define NC 32
#define NE (NR * NC)
#define NT 64            // NUMBER_THREADS (main.h): Rodinia uses 256, above the vl Hwacha can give these kernels
#define NITER 2
#define LAMBDA 0.5f
void extract_kernel_ct(long n, long Ne, float *I);
void prepare_kernel_ct(long n, long Ne, float *I, float *sums, float *sums2);
void reduce_kernel_ct(long n, long Ne, long no, int mul, float *sums, float *sums2, int gridDim);
void srad_kernel_ct(long n, float lambda, int Nr, int Nc, long Ne, int *iN, int *iS, int *jE, int *jW, float *dN, float *dS, float *dE, float *dW, float q0sqr, float *c, float *I);
void srad2_kernel_ct(long n, float lambda, int Nr, int Nc, long Ne, int *iN, int *iS, int *jE, int *jW, float *dN, float *dS, float *dE, float *dW, float *c, float *I);
void compress_kernel_ct(long n, long Ne, float *I);
static float I[NE], R[NE], sums[NE], sums2[NE], dN[NE], dS[NE], dE[NE], dW[NE], c[NE], rdN[NE], rdS[NE], rdE[NE], rdW[NE], rc[NE];
static int iN[NR], iS[NR], jE[NC], jW[NC];
int main(void) {
  for (int i = 0; i < NE; i++) I[i] = R[i] = frand(0, 255);
  for (int i = 0; i < NR; i++) { iN[i] = i - 1; iS[i] = i + 1; } iN[0] = 0; iS[NR-1] = NR - 1;
  for (int j = 0; j < NC; j++) { jW[j] = j - 1; jE[j] = j + 1; } jW[0] = 0; jE[NC-1] = NC - 1;
  unsigned long c0 = cyc();
  for (int i = 0; i < NE; i++) R[i] = expf(R[i] / 255);
  for (int it = 0; it < NITER; it++) {
    float total = 0, total2 = 0;
    for (int i = 0; i < NE; i++) { total += R[i]; total2 += R[i] * R[i]; }
    float mean = total / NE, var = total2 / NE - mean * mean, q0sqr = var / (mean * mean);
    for (int ei = 0; ei < NE; ei++) {
      int row = (ei + 1) % NR - 1, col = (ei + 1) / NR + 1 - 1; if ((ei + 1) % NR == 0) { row = NR - 1; col = col - 1; }
      float Jc = R[ei];
      float dn = R[iN[row] + NR*col] - Jc, ds = R[iS[row] + NR*col] - Jc, dw = R[row + NR*jW[col]] - Jc, de = R[row + NR*jE[col]] - Jc;
      float G2 = (dn*dn + ds*ds + dw*dw + de*de) / (Jc*Jc), L = (dn + ds + dw + de) / Jc;
      float num = (0.5*G2) - ((1.0/16.0)*(L*L)), den = 1 + (0.25*L), qsqr = num / (den*den);
      den = (qsqr - q0sqr) / (q0sqr * (1 + q0sqr)); float cl = 1.0 / (1.0 + den);
      if (cl < 0) cl = 0; else if (cl > 1) cl = 1;
      rdN[ei] = dn; rdS[ei] = ds; rdW[ei] = dw; rdE[ei] = de; rc[ei] = cl;
    }
    for (int ei = 0; ei < NE; ei++) {
      int row = (ei + 1) % NR - 1, col = (ei + 1) / NR + 1 - 1; if ((ei + 1) % NR == 0) { row = NR - 1; col = col - 1; }
      float D = rc[ei]*rdN[ei] + rc[iS[row] + NR*col]*rdS[ei] + rc[ei]*rdW[ei] + rc[row + NR*jE[col]]*rdE[ei];
      R[ei] = R[ei] + 0.25*LAMBDA*D;
    }
  }
  for (int i = 0; i < NE; i++) R[i] = logf(R[i]) * 255;
  unsigned long c1 = cyc(); REPORT("srad scalar", c0, c1, (long)NE*NITER);
  c0 = cyc();
  int blocks = CEILDIV(NE, NT);
  extract_kernel_ct(NDRANGE1(blocks * NT, NT), NE, I);
  for (int it = 0; it < NITER; it++) {
    prepare_kernel_ct(NDRANGE1(blocks * NT, NT), NE, I, sums, sums2);
    long no = NE; int mul = 1, blocks2 = blocks;
    while (blocks2 != 0) {
      reduce_kernel_ct(NDRANGE1(blocks2 * NT, NT), NE, no, mul, sums, sums2, blocks2);
      no = blocks2; mul *= NT;
      if (blocks2 == 1) blocks2 = 0; else blocks2 = CEILDIV(no, NT);
    }
    float mean = sums[0] / NE, var = sums2[0] / NE - mean * mean, q0sqr = var / (mean * mean);
    srad_kernel_ct(NDRANGE1(blocks * NT, NT), LAMBDA, NR, NC, NE, iN, iS, jE, jW, dN, dS, dE, dW, q0sqr, c, I);
    srad2_kernel_ct(NDRANGE1(blocks * NT, NT), LAMBDA, NR, NC, NE, iN, iS, jE, jW, dN, dS, dE, dW, c, I);
  }
  compress_kernel_ct(NDRANGE1(blocks * NT, NT), NE, I);
  c1 = cyc(); REPORT("srad hwacha-cc", c0, c1, (long)NE*NITER);
  int bad = check_f("image", I, R, NE, 1e-3f);
  printf("srad %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
