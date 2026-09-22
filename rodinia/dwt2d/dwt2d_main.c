// Rodinia dwt2d: one level of the forward 5/3 (reversible, integer lifting) 2-D DWT, cl_fdwt53Kernel
// with WIN_SX x WIN_SY sliding windows (a 2-D NDRange of WIN_SX-lane groups, as launchFDWT53Kernel
// does). The reference follows the kernel exactly: columns first (loadAndVerticallyTransform), then
// rows (forEachHorizontalOdd/Even), predict c -= (p + n) / 2 and update c += (p + n + 2) / 4 with C
// integer division, symmetric extension at the borders. Output layout (initialize_BandIO): four
// quadrant bands, each sx/2 wide and stored contiguously -- (even y, even x), (even y, odd x),
// (odd y, even x), (odd y, odd x) -- so coefficient (x, y) lands at
// out[((y & 1) * 2 + (x & 1)) * (sx * sy / 4) + (y / 2) * (sx / 2) + x / 2].
#include "common.h"
#define SX 64
#define SY 64
#define WIN_SX 32
#define WIN_SY 8
void cl_fdwt53Kernel_ct(long n, int *in, int *out, int sx, int sy, int steps, int WIN_SIZE_X, int WIN_SIZE_Y);
static int img[SX*SY], out[SX*SY], ref[SX*SY], tmp[SX*SY];
static int mir(int i, int n) { if (i < 0) i = -i; if (i >= n) i = 2 * n - 2 - i; return i; }   // symmetric extension
static void lift1d(const int *x, int n, int stride, int *lo, int *hi) {   // 5/3 forward lifting on one line
  int nh = n / 2, nl = n - nh; static int h[SX > SY ? SX : SY];
  for (int k = 0; k < nh; k++) h[k] = x[(2*k+1)*stride] - (x[mir(2*k, n)*stride] + x[mir(2*k+2, n)*stride]) / 2;
  for (int k = 0; k < nl; k++) { int hl = k > 0 ? h[k-1] : h[0], hr = k < nh ? h[k] : h[nh-1]; lo[k] = x[2*k*stride] + (hl + hr + 2) / 4; }
  for (int k = 0; k < nh; k++) hi[k] = h[k]; }
int main(void) {
  for (int i = 0; i < SX*SY; i++) img[i] = rnd() % 256;
  unsigned long c0 = cyc();
  { static int lo[SX > SY ? SX : SY], hi[SX > SY ? SX : SY], t2[SX*SY];
    for (int x = 0; x < SX; x++) { lift1d(img + x, SY, SX, lo, hi); for (int k = 0; k < SY/2; k++) { tmp[(2*k)*SX + x] = lo[k]; tmp[(2*k+1)*SX + x] = hi[k]; } }   // interleaved
    for (int y = 0; y < SY; y++) { lift1d(tmp + y*SX, SX, 1, lo, hi); for (int k = 0; k < SX/2; k++) { t2[y*SX + 2*k] = lo[k]; t2[y*SX + 2*k+1] = hi[k]; } }
    for (int y = 0; y < SY; y++) for (int x = 0; x < SX; x++) ref[((y & 1) * 2 + (x & 1)) * (SX*SY/4) + (y/2) * (SX/2) + x/2] = t2[y*SX + x]; }
  unsigned long c1 = cyc(); REPORT("dwt2d scalar", c0, c1, (long)SX*SY);
  int steps = SY / (15 * WIN_SY) + (SY % (15 * WIN_SY) ? 1 : 0);
  int gx = SX / WIN_SX + (SX % WIN_SX ? 1 : 0), gy = SY / (WIN_SY * steps) + (SY % (WIN_SY * steps) ? 1 : 0);
  c0 = cyc(); cl_fdwt53Kernel_ct(NDRANGE2(gx, gy, WIN_SX, 1), img, out, SX, SY, steps, WIN_SX, WIN_SY); c1 = cyc(); REPORT("dwt2d hwacha-cc", c0, c1, (long)SX*SY);
  { int nz = 0, nzin = 0; long sum = 0; for (int i = 0; i < SX*SY; i++) { if (out[i]) nz++; sum += out[i]; } for (int i = 0; i < SX*SY; i++) if (img[i] != (int)(0)) nzin++;
    printf("  out nonzero %d / %d, sum %ld; out[0..7] = %d %d %d %d %d %d %d %d; ref[0..7] = %d %d %d %d %d %d %d %d\n", nz, SX*SY, sum, out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], ref[0], ref[1], ref[2], ref[3], ref[4], ref[5], ref[6], ref[7]); }
  int bad = 0, first = -1; for (int i = 0; i < SX*SY; i++) if (out[i] != ref[i]) { if (first < 0) first = i; bad++; }
  printf("dwt2d: %d mismatches / %d%s\n", bad, SX*SY, hwacha_vl_short ? " (VL SHORT)" : ""); if (bad) printf("  first at %d (x=%d y=%d): hw %d ref %d\n", first, first % SX, first / SX, out[first], ref[first]);
  printf("dwt2d %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
