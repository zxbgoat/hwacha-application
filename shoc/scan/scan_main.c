// SHOC Scan (level1): inclusive prefix sum with the three SHOC kernels in Scan.cpp's order: reduce
// (each work-group sums its region into isums), top_scan (one work-group exclusive-scans isums),
// bottom_scan (each work-group scans its region 4 elements at a time, seeded by isums). __local
// buffers are host arrays (one work-group per stripmine). SHOC uses 64 work-groups of 256; here 64
// work-groups of 64 (bottom_scan's registers give maxvl 96). Input i % 3 as SHOC, exact compare.
#include "common.h"
#define N 16384
#define LS 64
#define GROUPS 64
void reduce_ct(long n, float *in, float *isums, int n_, float *lmem);
void top_scan_ct(long n, float *isums, int n_, float *lmem);
void bottom_scan_ct(long n, float *in, float *isums, float *out, int n_, float *lmem);
static float in[N], out[N], ref[N], isums[GROUPS], lmem[2*LS];
int main(void) {
  for (int i = 0; i < N; i++) in[i] = i % 3;
  unsigned long c0 = cyc(); float last = 0; for (int i = 0; i < N; i++) { ref[i] = in[i] + last; last = ref[i]; } unsigned long c1 = cyc();
  REPORT("scan scalar", c0, c1, N);
  c0 = cyc();
  reduce_ct(NDRANGE1(GROUPS * LS, LS), in, isums, N, lmem);
  top_scan_ct(NDRANGE1(LS, LS), isums, GROUPS, lmem);
  bottom_scan_ct(NDRANGE1(GROUPS * LS, LS), in, isums, out, N, lmem);
  c1 = cyc(); REPORT("scan hwacha-cc", c0, c1, N);
  int bad = 0; for (int i = 0; i < N; i++) if (out[i] != ref[i]) { if (bad < 4) printf("  mismatch at %d: hw %ld ref %ld\n", i, (long)out[i], (long)ref[i]); bad++; }
  printf("scan: %d mismatches / %d\n", bad, N);
  printf("scan %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
