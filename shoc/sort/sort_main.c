// SHOC Sort (level1): LSD radix sort of unsigned keys, 4-bit digits, 8 passes, with the three SHOC
// kernels per pass in Sort.cpp's order: reduce (per-group digit histograms into isums), top_scan
// (one work-group scans the 16 x groups histogram), bottom_scan (each work-group scatters its region
// 4 keys at a time), ping-ponging between two buffers. __local buffers are host arrays. SHOC uses 64
// work-groups of 256; here 64 of 32 for reduce / bottom_scan (bottom_scan's registers give maxvl 48; it
// needs >= 16 lanes) and one work-group of 64 for top_scan, which must span every group's histogram.
// Input i % 16 as SHOC; the check is verifySort's: the output is non-decreasing.
#include "common.h"
#define N 16384
#define LS 32
#define GROUPS 64
#define RADIX 4
void reduce_ct(long n, unsigned *in, unsigned *isums, int n_, unsigned *lmem, int shift);
void top_scan_ct(long n, unsigned *isums, int n_, unsigned *lmem);
void bottom_scan_ct(long n, unsigned *in, unsigned *isums, unsigned *out, int n_, unsigned *lmem, int shift);
static unsigned a[N], b[N], isums[16*GROUPS], lmem[2*GROUPS];
int main(void) {
  for (int i = 0; i < N; i++) a[i] = i % 16;
  unsigned long c0 = cyc();
  for (int shift = 0; shift < 32; shift += RADIX) {
    int even = (shift / RADIX) % 2 == 0; unsigned *in = even ? a : b, *out = even ? b : a;
    reduce_ct(NDRANGE1(GROUPS * LS, LS), in, isums, N, lmem, shift);
    top_scan_ct(NDRANGE1(GROUPS, GROUPS), isums, GROUPS, lmem);
    bottom_scan_ct(NDRANGE1(GROUPS * LS, LS), in, isums, out, N, lmem, shift);
  }
  unsigned long c1 = cyc(); REPORT("sort hwacha-cc", c0, c1, N);
  int bad = 0; for (int i = 1; i < N; i++) if (a[i-1] > a[i]) { if (bad < 4) printf("  out of order at %d: %u > %u\n", i, a[i-1], a[i]); bad++; }
  long sum = 0; for (int i = 0; i < N; i++) sum += a[i];
  printf("sort: %d out-of-order pairs / %d, key sum %ld (expected %ld)\n", bad, N - 1, sum, (long)N * 15 / 2);
  if (sum != (long)N * 15 / 2) bad++;
  printf("sort %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
