// SHOC Triad (level1): C = A + s*B over blocks of floats, the kernel Triad.cpp embeds. SHOC streams
// blocks of 64 KB .. 16 MB with s = 1.75 and random inputs in [0, 10); here one block per size for
// the three smallest block sizes (elements = blockSize*1024/4), work-groups of 128 as the SHOC host.
#include "common.h"
#define MAXN (256 * 1024 / 4)
#define LS 128
void Triad_ct(long n, const float *memA, const float *memB, float *memC, float s);
static float A[MAXN], B[MAXN], C[MAXN], ref[MAXN];
int main(void) {
  const float s = 1.75f; int bad = 0; static const int blockKB[] = {64, 128, 256};
  for (int i = 0; i < MAXN; i++) { A[i] = frand(0, 10); B[i] = frand(0, 10); }
  for (int b = 0; b < 3; b++) {
    int n = blockKB[b] * 1024 / 4;
    unsigned long c0 = cyc(); for (int i = 0; i < n; i++) ref[i] = A[i] + s * B[i]; unsigned long c1 = cyc(); unsigned long sc = c1 - c0;
    c0 = cyc(); Triad_ct(NDRANGE1(n, LS), A, B, C, s); c1 = cyc();
    printf("triad %d KB block (%d elements): scalar %lu cycles, hwacha-cc %lu cycles\n", blockKB[b], n, sc, c1 - c0);
    bad += check_f("  C", C, ref, n, 1e-6f);
  }
  printf("triad %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
