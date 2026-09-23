// PolyBench floyd-warshall (4.2.1): all-pairs shortest paths on an integer weight matrix. Kernel:
// floyd-warshall.cl, one 2-D launch per intermediate vertex k; integer result, compared exactly.
#include "common.h"
#define N 48
#define LX 8
#define LY 8
void floyd_warshall_kernel_ct(long n, int *path, int k, int n_);
static int path[N*N], ref[N*N];
static void ref_kernel(void) {
  for (int k = 0; k < N; k++) for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) ref[i*N + j] = ref[i*N + j] < ref[i*N + k] + ref[k*N + j] ? ref[i*N + j] : ref[i*N + k] + ref[k*N + j];
}
int main(void) {
  for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) { int v = i*j % 7 + 1; if ((i+j) % 13 == 0 || (i+j) % 7 == 0 || (i+j) % 11 == 0) v = 999; path[i*N + j] = ref[i*N + j] = v; }
  unsigned long c0 = cyc(); ref_kernel(); unsigned long c1 = cyc();
  REPORT("floyd-warshall scalar", c0, c1, (long)N*N*N);
  c0 = cyc();
  for (int k = 0; k < N; k++) floyd_warshall_kernel_ct(NDRANGE2(CEILDIV(N, LX), CEILDIV(N, LY), LX, LY), path, k, N);
  c1 = cyc(); REPORT("floyd-warshall hwacha-cc", c0, c1, (long)N*N*N);
  int bad = 0; for (int i = 0; i < N*N; i++) if (path[i] != ref[i]) bad++;
  printf("floyd-warshall: %d mismatches / %d\n", bad, N*N);
  printf("floyd-warshall %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
