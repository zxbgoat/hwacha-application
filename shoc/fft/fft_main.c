// SHOC FFT (level1): batched 512-point complex FFTs with the SHOC fft1D_512 / ifft1D_512 kernels
// (Volkov-style: a work-group of 64 work-items transforms one 512-point sequence, 8 points per
// work-item, radix-8 passes with twiddles and two transposes through __local memory) and the
// chk1D_512 kernel of SHOC's check. As FFT.cpp: random complex inputs in [-1, 1) with the second
// half of the batch a copy of the first, forward on the whole batch, inverse on the whole batch,
// then chk1D_512 compares the two halves. Here the forward result is also checked against a scalar
// (double) DFT and the round trip against the source (1e-4 relative: the kernel's twiddles come from
// hwacha-cc's sin / cos expansion). n_ffts scaled from 256 to 8.
#include "common.h"
#define NFFT 8
#define HALF (NFFT / 2)
#define LS 64
void fft1D_512_ct(long n, float *work);
void ifft1D_512_ct(long n, float *work);
void chk1D_512_ct(long n, float *work, int half_n_cmplx, int *fail);
static float src[NFFT*512*2], work[NFFT*512*2], ref[HALF*512*2];
int main(void) {
  for (int i = 0; i < HALF*512; i++) { float re = frand(-1, 1), im = frand(-1, 1); src[2*i] = re; src[2*i+1] = im; src[2*(i + HALF*512)] = re; src[2*(i + HALF*512)+1] = im; }
  unsigned long c0 = cyc();
  for (int f = 0; f < HALF; f++) for (int k = 0; k < 512; k++) { double sr = 0, si = 0; for (int n = 0; n < 512; n++) { double ang = -2.0 * 3.14159265358979323846 * (double)(k * n % 512) / 512.0, c = cos(ang), s = sin(ang); double xr = src[2*(f*512+n)], xi = src[2*(f*512+n)+1]; sr += xr*c - xi*s; si += xr*s + xi*c; } ref[2*(f*512+k)] = (float)sr; ref[2*(f*512+k)+1] = (float)si; }
  unsigned long c1 = cyc(); REPORT("fft scalar (DFT)", c0, c1, (long)HALF*512);
  for (int i = 0; i < NFFT*512*2; i++) work[i] = src[i];
  c0 = cyc(); fft1D_512_ct(NDRANGE1(NFFT * LS, LS), work); c1 = cyc(); REPORT("fft hwacha-cc forward", c0, c1, (long)NFFT*512);
  int bad = check_f("  forward vs DFT", work, ref, HALF*512*2, 1e-4f);
  c0 = cyc(); ifft1D_512_ct(NDRANGE1(NFFT * LS, LS), work); c1 = cyc(); REPORT("fft hwacha-cc inverse", c0, c1, (long)NFFT*512);
  bad += check_f("  round trip vs source", work, src, NFFT*512*2, 1e-4f);
  int fail = 0; chk1D_512_ct(NDRANGE1(HALF * LS, LS), work, HALF*512, &fail);
  printf("  chk1D_512 (two halves identical): %s\n", fail ? "FAIL" : "ok"); bad += fail;
  printf("fft %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
