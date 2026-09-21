// Rodinia lud: blocked in-place LU decomposition (no pivoting): per BLOCK_SIZE step the diagonal block
// (lud_diagonal), the perimeter blocks (lud_perimeter, 2*BS lanes) and the trailing sub-matrix
// (lud_internal, 2-D BS x BS tiles), as lud.cpp drives them; against a plain scalar LU
#include "common.h"
#define BS 8
#define DIM 64
void lud_diagonal_ct(long n, float *m, float *shadow, int matrix_dim, int offset);
void lud_perimeter_ct(long n, float *m, float *dia, float *peri_row, float *peri_col, int matrix_dim, int offset);
void lud_internal_ct(long n, float *m, float *peri_row, float *peri_col, int matrix_dim, int offset);
static float m[DIM*DIM], r[DIM*DIM], l0[BS*BS], l1[BS*BS], l2[BS*BS];
int main(void) {
  for (int i = 0; i < DIM*DIM; i++) m[i] = r[i] = frand(-1, 1);
  for (int i = 0; i < DIM; i++) m[i*DIM+i] = r[i*DIM+i] = frand(DIM, 2*DIM);   // diagonally dominant: no pivoting needed
  unsigned long c0 = cyc();
  for (int k = 0; k < DIM; k++) for (int i = k + 1; i < DIM; i++) {
    r[i*DIM+k] /= r[k*DIM+k];
    for (int j = k + 1; j < DIM; j++) r[i*DIM+j] -= r[i*DIM+k] * r[k*DIM+j];
  }
  unsigned long c1 = cyc(); REPORT("lud scalar", c0, c1, (long)DIM*DIM);
  c0 = cyc();
  int i;
  for (i = 0; i < DIM - BS; i += BS) {
    lud_diagonal_ct(NDRANGE1(BS, BS), m, l0, DIM, i);
    int nb = (DIM - i) / BS - 1;
    lud_perimeter_ct(NDRANGE1(2 * BS * nb, 2 * BS), m, l0, l1, l2, DIM, i);
    lud_internal_ct(NDRANGE2(nb, nb, BS, BS), m, l1, l2, DIM, i);
  }
  lud_diagonal_ct(NDRANGE1(BS, BS), m, l0, DIM, i);
  c1 = cyc(); REPORT("lud hwacha-cc", c0, c1, (long)DIM*DIM);
  int bad = check_f("lu", m, r, DIM*DIM, 1e-3f);
  printf("lud %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
