// SHOC Reduction (level1): sum of a float array with the SHOC `reduce` kernel (each work-group sums a
// strided slice into __local memory, tree-reduces it and writes one partial per group; the host adds
// the partials), launched as Reduction.cpp does: 64 work-groups of localWorkSize work-items, each
// work-item first accumulating pairs strided by the grid size. `reduceNoLocal` (SHOC's fallback for
// devices with work-group size 1) is run too. The __local buffer is a host array: every work-group
// runs on its own stripmine, so one buffer of localWorkSize elements serves all of them. Input i % 3
// as SHOC; check: |device sum - reference sum| < threshold (SHOC: 1e-8 in double; here 1e-3 relative,
// the partials are summed in a different order).
#include "common.h"
#define N 16384
#define LS 64
#define GROUPS 64
void reduce_ct(long n, float *g_idata, float *g_odata, float *sdata, unsigned n_);
void reduceNoLocal_ct(long n, float *g_idata, float *g_odata, unsigned n_);
static float idata[N], odata[GROUPS], sdata[LS], odata1[1];
int main(void) {
  for (int i = 0; i < N; i++) idata[i] = i % 3;
  unsigned long c0 = cyc(); double ref = 0; for (int i = 0; i < N; i++) ref += idata[i]; unsigned long c1 = cyc();
  REPORT("reduction scalar", c0, c1, N);
  c0 = cyc(); reduce_ct(NDRANGE1(GROUPS * LS, LS), idata, odata, sdata, N); c1 = cyc();
  REPORT("reduction hwacha-cc", c0, c1, N);
  double dev = 0; for (int i = 0; i < GROUPS; i++) dev += odata[i];
  unsigned long c2 = cyc(); reduceNoLocal_ct(NDRANGE1(1, 1), idata, odata1, N); unsigned long c3 = cyc();
  REPORT("reduction hwacha-cc reduceNoLocal", c2, c3, N);
  int bad = fabs(dev - ref) > 1e-3 * ref; int bad1 = fabs((double)odata1[0] - ref) > 1e-3 * ref;
  printf("reduction: reduce sum %ld ref %ld (x1000), reduceNoLocal sum %ld -> %s\n", (long)(dev * 1000), (long)(ref * 1000), (long)(odata1[0] * 1000.0), (bad || bad1) ? "FAIL" : "ok");
  printf("reduction %s\n", (bad || bad1) ? "FAIL" : "PASS"); return bad || bad1;
}
