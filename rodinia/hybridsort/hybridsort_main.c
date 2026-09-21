// Rodinia hybridsort, bucket stage (histogram1024 + bucketsort_kernels; the merge stage's mergesort.cl
// uses float4 vector ops hwacha-cc does not take): histogram of the input (warp-tagged __local
// atomics), pivot points on the host (calcPivotPoints), bucketcount (per-element bucket id + slot),
// bucketprefixoffset, host-side bucket starts, bucketsort (scatter). Launched as bucketsort.c does.
// The slot an element gets inside its bucket depends on which lane wins a same-bucket conflict, so
// the check compares each bucket's *contents* (sorted) with the elements the scalar port assigns to it.
#include "common.h"
#define N 2048
#define DIVISIONS 1024
#define LOG_DIVISIONS 10
#define BUCKET_THREAD_N 32
#define BUCKET_BAND 128
#define BUCKET_BLOCK_MEMORY DIVISIONS
#define BIN_COUNT 1024
#define WARP_N 3
#define HISTO_LOCAL 96
#define HISTO_GLOBAL 6144
void histogram1024Kernel_ct(long n, unsigned *result, float *data, float minimum, float maximum, unsigned dataCount);
void bucketcount_ct(long n, float *input, int *indice, unsigned *prefixoffsets, int size, float *pivotpoints);
void bucketprefixoffset_ct(long n, unsigned *prefixoffsets, unsigned *offsets, int blocks);
void bucketsort_ct(long n, float *input, int *indice, float *output, int size, unsigned *prefixoffsets, unsigned *offsets);
#define BLOCKS (((N - 1) / (BUCKET_THREAD_N * BUCKET_BAND)) + 1)
static float in[N + DIVISIONS * 4], out[N + DIVISIONS * 4], piv[DIVISIONS], histo[BIN_COUNT]; static unsigned hist[BIN_COUNT], rhist[BIN_COUNT], pre[BLOCKS * BUCKET_BLOCK_MEMORY], offs[DIVISIONS], roffs[DIVISIONS], starts[DIVISIONS]; static int indice[N], rbucket[N], cnt[DIVISIONS];
static void calcPivotPoints(float *histogram, int histosize, int listsize, int divisions, float min, float max, float *pivotPoints, float histo_width) {
  float elemsPerSlice = listsize / (float)divisions, startsAt = min, endsAt = min + histo_width, we_need = elemsPerSlice; int p_idx = 0;
  for (int i = 0; i < histosize; i++) {
    if (i == histosize - 1) { if (!(p_idx < divisions)) pivotPoints[p_idx++] = startsAt + (we_need / histogram[i]) * histo_width; break; }
    while (histogram[i] > we_need) { if (!(p_idx < divisions)) break; pivotPoints[p_idx++] = startsAt + (we_need / histogram[i]) * histo_width; startsAt += (we_need / histogram[i]) * histo_width; histogram[i] -= we_need; we_need = elemsPerSlice; }
    we_need -= histogram[i]; startsAt = endsAt; endsAt += histo_width; }
  for (int i = p_idx; i < divisions; i++) pivotPoints[i] = max; }
static int cmpf(const void *a, const void *b) { float x = *(const float *)a, y = *(const float *)b; return x < y ? -1 : x > y; }
static void sortf(float *a, int n) { for (int i = 1; i < n; i++) { float v = a[i]; int j = i - 1; while (j >= 0 && a[j] > v) { a[j+1] = a[j]; j--; } a[j+1] = v; } }
int main(void) {
  float mn = 0, mx = 100; for (int i = 0; i < N; i++) in[i] = frand(mn, mx); for (int i = N; i < N + DIVISIONS * 4; i++) in[i] = 0;
  unsigned long c0 = cyc();
  for (int i = 0; i < N; i++) { unsigned b = (unsigned)(((in[i] - mn) / (mx - mn)) * BIN_COUNT) & 0x3FF; rhist[b]++; }
  for (int i = 0; i < BIN_COUNT; i++) histo[i] = (float)rhist[i];
  calcPivotPoints(histo, BIN_COUNT, N, DIVISIONS, mn, mx, piv, (mx - mn) / (float)BIN_COUNT);
  for (int i = 0; i < N; i++) { float e = in[i]; int idx = DIVISIONS/2 - 1, jump = DIVISIONS/4; float p = piv[idx]; while (jump >= 1) { idx = e < p ? idx - jump : idx + jump; p = piv[idx]; jump /= 2; } idx = e < p ? idx : idx + 1; rbucket[i] = idx; cnt[idx]++; }
  for (int i = 0; i < DIVISIONS; i++) roffs[i] = cnt[i];
  unsigned long c1 = cyc(); REPORT("hybridsort scalar", c0, c1, (long)N);
  c0 = cyc();
  histogram1024Kernel_ct(NDRANGE1(HISTO_GLOBAL, HISTO_LOCAL), hist, in, mn, mx, N);
  // (the pivots come from the histogram, on the host, as bucketSort() does)
  for (int i = 0; i < BIN_COUNT; i++) histo[i] = (float)hist[i];
  calcPivotPoints(histo, BIN_COUNT, N, DIVISIONS, mn, mx, piv, (mx - mn) / (float)BIN_COUNT);
  bucketcount_ct(NDRANGE1(BLOCKS * BUCKET_THREAD_N, BUCKET_THREAD_N), in, indice, pre, N, piv);
  bucketprefixoffset_ct(NDRANGE1(DIVISIONS, 128), pre, offs, BLOCKS);
  { unsigned h[DIVISIONS]; for (int i = 0; i < DIVISIONS; i++) h[i] = offs[i] % 4 ? (offs[i] & ~3u) + 4 : offs[i];   // bucketSort(): pad to 4, exclusive prefix
    for (int i = 1; i < DIVISIONS; i++) h[i] = h[i-1] + h[i]; for (int i = DIVISIONS - 1; i > 0; i--) h[i] = h[i-1]; h[0] = 0; memcpy(starts, h, sizeof h); }
  bucketsort_ct(NDRANGE1(BLOCKS * BUCKET_THREAD_N, BUCKET_THREAD_N), in, indice, out, N, pre, starts);
  c1 = cyc(); REPORT("hybridsort hwacha-cc", c0, c1, (long)N);
  int bad = 0; for (int i = 0; i < BIN_COUNT; i++) if (hist[i] != rhist[i]) bad++; printf("histogram: %d mismatched bins / %d\n", bad, BIN_COUNT);
  int badc = 0; for (int i = 0; i < DIVISIONS; i++) if (offs[i] != roffs[i]) badc++; printf("bucket counts: %d mismatches / %d\n", badc, DIVISIONS); bad += badc;
  int badb = 0; static float hw[N], rf[N];
  for (int b = 0; b < DIVISIONS; b++) { int n = cnt[b]; if (!n) continue; int k = 0; for (int i = 0; i < N; i++) if (rbucket[i] == b) rf[k++] = in[i]; for (int i = 0; i < n; i++) hw[i] = out[starts[b] + i]; sortf(hw, n); sortf(rf, n); for (int i = 0; i < n; i++) if (hw[i] != rf[i]) { badb++; break; } }
  printf("bucket contents: %d buckets differ / %d%s\n", badb, DIVISIONS, hwacha_vl_short ? " (VL SHORT)" : ""); bad += badb;
  printf("hybridsort %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
