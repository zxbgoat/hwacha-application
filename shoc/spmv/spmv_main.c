// SHOC Spmv (level1): y = A x with the three SHOC kernels -- spmv_csr_scalar_kernel (one work-item per
// row), spmv_csr_vector_kernel (vecWidth work-items per row, partial sums tree-reduced in __local
// memory) and spmv_ellpackr_kernel (column-major ELLPACK-R). As Spmv.cpp: a random matrix with 1% of
// the entries non-zero (util.h initRandomMatrix: each entry kept with probability nnz / dim^2, the
// tail filled so the count is exact), values and x uniform in [0, maxval = 10), reference spmvCpu,
// per-row relative error check. numRows scaled from 1024 to 256; BLOCK_SIZE 128, vecWidth 16.
#include "common.h"
#define DIM 256
#define NNZ (DIM * DIM / 100)
#define LS 128
#define VW 16
void spmv_csr_scalar_kernel_ct(long n, float *val, float *vec, int *cols, int *rowDelim, int dim, float *out);
void spmv_csr_vector_kernel_ct(long n, float *val, float *vec, int *cols, int *rowDelim, int dim, int vecWidth, float *out);
void spmv_ellpackr_kernel_ct(long n, float *val, float *vec, int *cols, int *rowLengths, int dim, float *out);
static float val[NNZ], vec[DIM], out[DIM], ref[DIM], valcm[DIM*DIM]; static int cols[NNZ], rowDelim[DIM+1], colscm[DIM*DIM], rl[DIM];
static int check(const char *tag) { int bad = 0; for (int i = 0; i < DIM; i++) if (fabsf(ref[i] - out[i]) / fabsf(ref[i]) > 1e-4f) bad++; printf("  %s: %d rows above 1e-4 relative / %d\n", tag, bad, DIM); return bad; }
int main(void) {
  // initRandomMatrix
  int nnz = 0; double prob = (double)NNZ / ((double)DIM * DIM); int fillRemaining = 0;
  for (int i = 0; i < DIM; i++) { rowDelim[i] = nnz; for (int j = 0; j < DIM; j++) { int left = DIM*DIM - (i*DIM + j), need = NNZ - nnz; if (left <= need) fillRemaining = 1; if ((nnz < NNZ && frand(0, 1) <= prob) || fillRemaining) { cols[nnz++] = j; } } }
  rowDelim[DIM] = NNZ;
  for (int i = 0; i < NNZ; i++) val[i] = frand(0, 10);
  for (int i = 0; i < DIM; i++) vec[i] = frand(0, 10);
  unsigned long c0 = cyc();
  for (int i = 0; i < DIM; i++) { float t = 0; for (int j = rowDelim[i]; j < rowDelim[i+1]; j++) t += val[j] * vec[cols[j]]; ref[i] = t; }
  unsigned long c1 = cyc(); REPORT("spmv scalar", c0, c1, NNZ);
  int bad = 0;
  c0 = cyc(); spmv_csr_scalar_kernel_ct(NDRANGE1(DIM, LS), val, vec, cols, rowDelim, DIM, out); c1 = cyc();
  REPORT("spmv hwacha-cc csr_scalar", c0, c1, NNZ); bad += check("csr_scalar");
  for (int i = 0; i < DIM; i++) out[i] = 0;
  c0 = cyc(); spmv_csr_vector_kernel_ct(NDRANGE1(DIM * VW, LS), val, vec, cols, rowDelim, DIM, VW, out); c1 = cyc();
  REPORT("spmv hwacha-cc csr_vector", c0, c1, NNZ); bad += check("csr_vector");
  // convertToColMajor (ELLPACK-R): maxrl = longest row
  int maxrl = 0; for (int i = 0; i < DIM; i++) { rl[i] = rowDelim[i+1] - rowDelim[i]; if (rl[i] > maxrl) maxrl = rl[i]; }
  int k = 0; for (int j = 0; j < maxrl; j++) for (int i = 0; i < DIM; i++) { if (rowDelim[i] + j < rowDelim[i+1]) { valcm[k] = val[rowDelim[i] + j]; colscm[k] = cols[rowDelim[i] + j]; } else valcm[k] = 0; k++; }
  for (int i = 0; i < DIM; i++) out[i] = 0;
  c0 = cyc(); spmv_ellpackr_kernel_ct(NDRANGE1(DIM, LS), valcm, vec, colscm, rl, DIM, out); c1 = cyc();
  REPORT("spmv hwacha-cc ellpackr", c0, c1, NNZ); bad += check("ellpackr");
  printf("spmv %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
