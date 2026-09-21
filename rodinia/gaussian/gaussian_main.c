// Rodinia gaussian: Gaussian elimination of an NxN system, Fan1 (multipliers, 1-D) + Fan2 (row
// updates, 2-D) per pivot, driven from the host like gaussianElim.cpp; checked against a scalar copy
#include "common.h"
#define N 64
#define LS 8
void Fan1_ct(long n, float *m, float *a, float *b, int size, int t);
void Fan2_ct(long n, float *m, float *a, float *b, int size, int t);
static float a[N*N], b[N], m[N*N], ra[N*N], rb[N], rm[N*N];
int main(void) {
  for (int i = 0; i < N*N; i++) a[i] = ra[i] = frand(-1, 1);
  for (int i = 0; i < N; i++) { a[i*N+i] = ra[i*N+i] = frand(4, 8); b[i] = rb[i] = frand(-1, 1); }   // diagonally dominant
  unsigned long c0 = cyc();
  for (int t = 0; t < N - 1; t++) {
    for (int i = 0; i < N - 1 - t; i++) rm[N*(i+t+1)+t] = ra[N*(i+t+1)+t] / ra[N*t+t];
    for (int i = 0; i < N - 1 - t; i++) for (int j = 0; j < N - t; j++) {
      ra[N*(i+1+t)+(j+t)] -= rm[N*(i+1+t)+t] * ra[N*t+(j+t)];
      if (j == 0) rb[i+1+t] -= rm[N*(i+1+t)+(j+t)] * rb[t];
    }
  }
  unsigned long c1 = cyc(); REPORT("gaussian scalar", c0, c1, N*N);
  c0 = cyc();
  for (int t = 0; t < N - 1; t++) {
    Fan1_ct(NDRANGE1(N, 0), m, a, b, N, t);
    Fan2_ct(NDRANGE2(CEILDIV(N - 1 - t, LS), CEILDIV(N - t, LS), LS, LS), m, a, b, N, t);
  }
  c1 = cyc(); REPORT("gaussian hwacha-cc", c0, c1, N*N);
  int bad = check_f("a", a, ra, N*N, 1e-4f) + check_f("b", b, rb, N, 1e-4f) + check_f("m", m, rm, N*N, 1e-4f);
  printf("gaussian %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
