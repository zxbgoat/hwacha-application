// Rodinia myocyte: one evaluation of the cardiac myocyte ODE right-hand side (kernel_gpu_opencl:
// group 0 / lane 0 runs the ECC model, group 1 / lane 0 the three CaM models), i.e. one call of the
// Fehlberg solver's derivative function; the reference is the same kernel code compiled for the scalar
// core (myocyte_ref.c, the .cl with the OpenCL qualifiers defined away)
#include "common.h"
#define EQUATIONS 91
#define PARAMETERS 18
void kernel_gpu_opencl_ct(long n, int timeinst, float *initvalu, float *finavalu, float *params, float *com);
void ref_kernel(int timeinst, float *initvalu, float *finavalu, float *params, float *com, int bx, int tx);
static float initvalu[EQUATIONS], finavalu[EQUATIONS], rfinavalu[EQUATIONS], params[PARAMETERS], com[3], rcom[3];
int main(void) {
  // initial state and parameters in the ranges main.c uses (y[0][k] = 0..1-ish, params 0..1)
  for (int k = 0; k < EQUATIONS; k++) { initvalu[k] = frand(0.01f, 1); finavalu[k] = rfinavalu[k] = 0; }
  for (int k = 0; k < PARAMETERS; k++) params[k] = frand(0.1f, 1);
  unsigned long c0 = cyc(); ref_kernel(0, initvalu, rfinavalu, params, rcom, 0, 0); ref_kernel(0, initvalu, rfinavalu, params, rcom, 1, 0); unsigned long c1 = cyc(); REPORT("myocyte scalar", c0, c1, (long)EQUATIONS);
  c0 = cyc(); kernel_gpu_opencl_ct(NDRANGE1(2 * 2, 2), 0, initvalu, finavalu, params, com); c1 = cyc(); REPORT("myocyte hwacha-cc", c0, c1, (long)EQUATIONS);
  int bad = check_f("finavalu", finavalu, rfinavalu, EQUATIONS, 1e-3f);
  printf("myocyte %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
