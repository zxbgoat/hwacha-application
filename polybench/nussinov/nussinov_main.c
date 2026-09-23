// PolyBench nussinov (4.2.1): Nussinov RNA secondary-structure score table (integer dynamic
// programming). Kernel: nussinov.cl, one launch per diagonal d = j - i (entries on a diagonal
// depend only on shorter ones), one work-item per entry; compared exactly.
#include "common.h"
#define N 48
#define LS 64
void nussinov_kernel_ct(long n, char *seq, int *table, int d, int n_);
static char seq[N]; static int table[N*N], ref[N*N];
#define match(b1, b2) (((b1)+(b2)) == 3 ? 1 : 0)
#define max_score(s1, s2) ((s1 >= s2) ? s1 : s2)
static void ref_kernel(void) {
  for (int i = N-1; i >= 0; i--) for (int j = i+1; j < N; j++) {
    if (j-1 >= 0) ref[i*N + j] = max_score(ref[i*N + j], ref[i*N + (j-1)]);
    if (i+1 < N) ref[i*N + j] = max_score(ref[i*N + j], ref[(i+1)*N + j]);
    if (j-1 >= 0 && i+1 < N) { if (i < j-1) ref[i*N + j] = max_score(ref[i*N + j], ref[(i+1)*N + (j-1)] + match(seq[i], seq[j])); else ref[i*N + j] = max_score(ref[i*N + j], ref[(i+1)*N + (j-1)]); }
    for (int k = i+1; k < j; k++) ref[i*N + j] = max_score(ref[i*N + j], ref[i*N + k] + ref[(k+1)*N + j]);
  }
}
int main(void) {
  for (int i = 0; i < N; i++) seq[i] = (char)((i+1) % 4);
  for (int i = 0; i < N*N; i++) table[i] = ref[i] = 0;
  unsigned long c0 = cyc(); ref_kernel(); unsigned long c1 = cyc();
  REPORT("nussinov scalar", c0, c1, (long)N*N*N/6);
  c0 = cyc();
  for (int d = 1; d < N; d++) nussinov_kernel_ct(NDRANGE1(CEILDIV(N - d, LS)*LS, LS), seq, table, d, N);
  c1 = cyc(); REPORT("nussinov hwacha-cc", c0, c1, (long)N*N*N/6);
  int bad = 0; for (int i = 0; i < N*N; i++) if (table[i] != ref[i]) bad++;
  printf("nussinov: %d mismatches / %d (score[0][N-1] = %d)\n", bad, N*N, table[N-1]);
  printf("nussinov %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
