// Rodinia particlefilter (naive OpenCL version): the resampling step -- for every particle i, find the
// first CDF entry >= u[i] (linear scan, as the kernel does) and copy that particle's state; doubles
// throughout, as the kernel. CDF is a sorted cumulative-weight vector, u the stratified samples
// u[i] = u1 + i/N as particleFilter() builds them; reference = the same scan on the scalar core
#include "common.h"
#define NP 512
void particle_kernel_ct(long n, double *arrayX, double *arrayY, double *CDF, double *u, double *xj, double *yj, int Nparticles);
static double ax[NP], ay[NP], cdf[NP], u[NP], xj[NP], yj[NP], rxj[NP], ryj[NP];
int main(void) {
  double acc = 0;
  for (int i = 0; i < NP; i++) { ax[i] = frand(-10, 10); ay[i] = frand(-10, 10); acc += frand(0.01f, 1); cdf[i] = acc; }
  for (int i = 0; i < NP; i++) cdf[i] /= acc;
  double u1 = frand(0, 1) / NP; for (int i = 0; i < NP; i++) u[i] = u1 + i / (double)NP;
  unsigned long c0 = cyc();
  for (int i = 0; i < NP; i++) { int idx = -1; for (int x = 0; x < NP; x++) if (cdf[x] >= u[i]) { idx = x; break; } if (idx == -1) idx = NP - 1; rxj[i] = ax[idx]; ryj[i] = ay[idx]; }
  unsigned long c1 = cyc(); REPORT("particlefilter scalar", c0, c1, (long)NP);
  c0 = cyc(); particle_kernel_ct(NDRANGE1(NP, 128), ax, ay, cdf, u, xj, yj, NP); c1 = cyc(); REPORT("particlefilter hwacha-cc", c0, c1, (long)NP);
  int bad = 0; for (int i = 0; i < NP; i++) if (xj[i] != rxj[i] || yj[i] != ryj[i]) bad++;
  printf("particlefilter: %d mismatches / %d\n", bad, NP); printf("particlefilter %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
