// Rodinia nw: Needleman-Wunsch DP over a (N+1)x(N+1) score matrix, processed by anti-diagonals of
// BLOCK_SIZE x BLOCK_SIZE blocks (nw_kernel1 for the upper-left triangle, nw_kernel2 for the rest),
// as nw.c drives it; against the plain scalar DP
#include "common.h"
#define BS 16
#define N 64            // sequence length; the matrices are (N+1) x (N+1)
#define PEN 10
#define MC (N + 1)
void nw_kernel1_ct(long n, int *reference_d, int *input_d, int *output_d, int *input_l, int *reference_l, int cols, int penalty, int blk, int block_width, int worksize, int offset_r, int offset_c);
void nw_kernel2_ct(long n, int *reference_d, int *input_d, int *output_d, int *input_l, int *reference_l, int cols, int penalty, int blk, int block_width, int worksize, int offset_r, int offset_c);
static int ref[MC*MC], in[MC*MC], rr[MC*MC], out[MC*MC], l1[(BS+1)*(BS+1)], l2[BS*BS];
static int max3(int a, int b, int c) { int k = a <= b ? b : a; return k <= c ? c : k; }
int main(void) {
  for (int i = 1; i < MC; i++) for (int j = 1; j < MC; j++) ref[i*MC+j] = (int)(rnd() % 16) - 4;   // stands in for blosum62[seq1][seq2]
  for (int i = 1; i < MC; i++) { in[i*MC] = rr[i*MC] = -i * PEN; in[i] = rr[i] = -i * PEN; }
  unsigned long c0 = cyc();
  for (int i = 1; i < MC; i++) for (int j = 1; j < MC; j++)
    rr[i*MC+j] = max3(rr[(i-1)*MC+j-1] + ref[i*MC+j], rr[i*MC+j-1] - PEN, rr[(i-1)*MC+j] - PEN);
  unsigned long c1 = cyc(); REPORT("nw scalar", c0, c1, (long)N*N);
  c0 = cyc();
  int worksize = MC - 1, block_width = worksize / BS;
  for (int blk = 1; blk <= block_width; blk++) nw_kernel1_ct(NDRANGE1(BS * blk, BS), ref, in, out, l1, l2, MC, PEN, blk, block_width, worksize, 0, 0);
  for (int blk = block_width - 1; blk >= 1; blk--) nw_kernel2_ct(NDRANGE1(BS * blk, BS), ref, in, out, l1, l2, MC, PEN, blk, block_width, worksize, 0, 0);
  c1 = cyc(); REPORT("nw hwacha-cc", c0, c1, (long)N*N);
  int bad = 0; for (int i = 0; i < MC*MC; i++) if (in[i] != rr[i]) bad++;
  printf("nw: %d mismatches / %d%s\n", bad, MC*MC, hwacha_vl_short ? " (VL SHORT)" : "");
  printf("nw %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
