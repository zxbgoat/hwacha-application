// Rodinia cfd (euler3d): the five kernels over a small synthetic mesh, driven exactly like
// euler3d.cpp's main loop (copy, step factor, RK stages of flux + time step); reference = a scalar
// port of the same kernels over the same mesh
#include "common.h"
#define NEL 256
#define NELR NEL
#define NDIM 3
#define NNB 4
#define RK 3
#define GAMMA 1.4f
#define ITER 2
#define VAR_DENSITY 0
#define VAR_MOMENTUM 1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM + NDIM)
#define NVAR (VAR_DENSITY_ENERGY + 1)
typedef struct { float x, y, z; } FLOAT3;
void memset_kernel_ct(long n, char *mem, short val, int ct);
void initialize_variables_ct(long n, float *variables, float *ff_variable, int nelr);
void compute_step_factor_ct(long n, float *variables, float *areas, float *step_factors, int nelr);
void compute_flux_ct(long n, int *ese, float *normals, float *variables, float *ff_variable, float *fluxes, FLOAT3 *ffde, FLOAT3 *ffmx, FLOAT3 *ffmy, FLOAT3 *ffmz, int nelr);
void time_step_ct(long n, int j, int nelr, float *old_variables, float *variables, float *step_factors, float *fluxes);
static float areas[NELR], normals[NELR*NNB*NDIM], variables[NELR*NVAR], old_variables[NELR*NVAR], step_factors[NELR], fluxes[NELR*NVAR];
static float rv[NELR*NVAR], rold[NELR*NVAR], rsf[NELR], rfl[NELR*NVAR];
static int ese[NELR*NNB]; static float ff_variable[NVAR]; static FLOAT3 ffde, ffmx, ffmy, ffmz;
// ---- scalar port of the kernels (same arithmetic, same order)
static void velocity(float d, FLOAT3 m, FLOAT3 *v) { v->x = m.x / d; v->y = m.y / d; v->z = m.z / d; }
static float speed_sqd(FLOAT3 v) { return v.x*v.x + v.y*v.y + v.z*v.z; }
static float pressure(float d, float de, float ss) { return (GAMMA - 1.0f) * (de - 0.5f * d * ss); }
static float sos(float d, float p) { return sqrtf(GAMMA * p / d); }
static void fluxc(float d, FLOAT3 m, float de, float p, FLOAT3 v, FLOAT3 *fx, FLOAT3 *fy, FLOAT3 *fz, FLOAT3 *fde) {
  fx->x = v.x*m.x + p; fx->y = v.x*m.y; fx->z = v.x*m.z; fy->x = fx->y; fy->y = v.y*m.y + p; fy->z = v.y*m.z; fz->x = fx->z; fz->y = fy->z; fz->z = v.z*m.z + p;
  float dp = de + p; fde->x = v.x*dp; fde->y = v.y*dp; fde->z = v.z*dp; }
static void ref_step_factor(float *var, float *sf) {
  for (int i = 0; i < NELR; i++) { float d = var[i + VAR_DENSITY*NELR]; FLOAT3 m = {var[i + 1*NELR], var[i + 2*NELR], var[i + 3*NELR]}; float de = var[i + VAR_DENSITY_ENERGY*NELR];
    FLOAT3 v; velocity(d, m, &v); float ss = speed_sqd(v); float p = pressure(d, de, ss); sf[i] = 0.5f / (sqrtf(areas[i]) * (sqrtf(ss) + sos(d, p))); } }
static void ref_flux(float *var, float *fl) {
  for (int i = 0; i < NELR; i++) {
    float di = var[i + VAR_DENSITY*NELR]; FLOAT3 mi = {var[i + 1*NELR], var[i + 2*NELR], var[i + 3*NELR]}; float dei = var[i + VAR_DENSITY_ENERGY*NELR];
    FLOAT3 vi; velocity(di, mi, &vi); float ssi = speed_sqd(vi), spi = sqrtf(ssi), pi = pressure(di, dei, ssi), sosi = sos(di, pi);
    FLOAT3 fix, fiy, fiz, fide; fluxc(di, mi, dei, pi, vi, &fix, &fiy, &fiz, &fide);
    float fd = 0, fde = 0; FLOAT3 fm = {0, 0, 0};
    for (int j = 0; j < NNB; j++) {
      int nb = ese[i + j*NELR]; FLOAT3 n = {normals[i + (j + 0*NNB)*NELR], normals[i + (j + 1*NNB)*NELR], normals[i + (j + 2*NNB)*NELR]}; float nl = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z); float f;
      if (nb >= 0) {
        float dn = var[nb + VAR_DENSITY*NELR]; FLOAT3 mn = {var[nb + 1*NELR], var[nb + 2*NELR], var[nb + 3*NELR]}; float den = var[nb + VAR_DENSITY_ENERGY*NELR];
        FLOAT3 vn; velocity(dn, mn, &vn); float ssn = speed_sqd(vn), pn = pressure(dn, den, ssn), sosn = sos(dn, pn); FLOAT3 fnx, fny, fnz, fnde; fluxc(dn, mn, den, pn, vn, &fnx, &fny, &fnz, &fnde);
        f = -nl * 0.2f * 0.5f * (spi + sqrtf(ssn) + sosi + sosn);
        fd += f*(di - dn); fde += f*(dei - den); fm.x += f*(mi.x - mn.x); fm.y += f*(mi.y - mn.y); fm.z += f*(mi.z - mn.z);
        f = 0.5f*n.x; fd += f*(mn.x + mi.x); fde += f*(fnde.x + fide.x); fm.x += f*(fnx.x + fix.x); fm.y += f*(fny.x + fiy.x); fm.z += f*(fnz.x + fiz.x);
        f = 0.5f*n.y; fd += f*(mn.y + mi.y); fde += f*(fnde.y + fide.y); fm.x += f*(fnx.y + fix.y); fm.y += f*(fny.y + fiy.y); fm.z += f*(fnz.y + fiz.y);
        f = 0.5f*n.z; fd += f*(mn.z + mi.z); fde += f*(fnde.z + fide.z); fm.x += f*(fnx.z + fix.z); fm.y += f*(fny.z + fiy.z); fm.z += f*(fnz.z + fiz.z);
      } else if (nb == -1) { fm.x += n.x*pi; fm.y += n.y*pi; fm.z += n.z*pi; }
      else if (nb == -2) {
        f = 0.5f*n.x; fd += f*(ff_variable[VAR_MOMENTUM+0] + mi.x); fde += f*(ffde.x + fide.x); fm.x += f*(ffmx.x + fix.x); fm.y += f*(ffmy.x + fiy.x); fm.z += f*(ffmz.x + fiz.x);
        f = 0.5f*n.y; fd += f*(ff_variable[VAR_MOMENTUM+1] + mi.y); fde += f*(ffde.y + fide.y); fm.x += f*(ffmx.y + fix.y); fm.y += f*(ffmy.y + fiy.y); fm.z += f*(ffmz.y + fiz.y);
        f = 0.5f*n.z; fd += f*(ff_variable[VAR_MOMENTUM+2] + mi.z); fde += f*(ffde.z + fide.z); fm.x += f*(ffmx.z + fix.z); fm.y += f*(ffmy.z + fiy.z); fm.z += f*(ffmz.z + fiz.z);
      }
    }
    fl[i + VAR_DENSITY*NELR] = fd; fl[i + 1*NELR] = fm.x; fl[i + 2*NELR] = fm.y; fl[i + 3*NELR] = fm.z; fl[i + VAR_DENSITY_ENERGY*NELR] = fde;
  } }
static void ref_time_step(int j, float *old, float *var, float *sf, float *fl) {
  for (int i = 0; i < NELR; i++) { float f = sf[i] / (float)(RK + 1 - j); for (int v = 0; v < NVAR; v++) var[i + v*NELR] = old[i + v*NELR] + f * fl[i + v*NELR]; } }
int main(void) {
  // far-field state (euler3d.cpp) and a synthetic mesh: random areas / normals, 4 neighbours per element
  // (a ring: i-1, i+1, and two random ones), 12 elements with a wing (-1) and 12 with a far-field (-2) face
  float ffp = 1.0f, ffsos = sqrtf(GAMMA * ffp / 1.4f), ffs = 1.2f * ffsos; ff_variable[VAR_DENSITY] = 1.4f;
  FLOAT3 ffv = {ffs, 0, 0}; ff_variable[VAR_MOMENTUM+0] = 1.4f*ffv.x; ff_variable[VAR_MOMENTUM+1] = 0; ff_variable[VAR_MOMENTUM+2] = 0;
  ff_variable[VAR_DENSITY_ENERGY] = 1.4f * (0.5f * ffs * ffs) + ffp / (GAMMA - 1.0f);
  FLOAT3 ffm = {ff_variable[1], ff_variable[2], ff_variable[3]}; fluxc(ff_variable[VAR_DENSITY], ffm, ff_variable[VAR_DENSITY_ENERGY], ffp, ffv, &ffmx, &ffmy, &ffmz, &ffde);
  for (int i = 0; i < NELR; i++) { areas[i] = frand(0.5f, 1.5f);
    for (int j = 0; j < NNB; j++) { int nb = j == 0 ? (i + NELR - 1) % NELR : j == 1 ? (i + 1) % NELR : (int)(rnd() % NELR); if (j == 2 && i % 21 == 0) nb = -1; if (j == 3 && i % 21 == 10) nb = -2; ese[i + j*NELR] = nb;
      for (int k = 0; k < NDIM; k++) normals[i + (j + k*NNB)*NELR] = frand(-1, 1); } }
  // reference
  unsigned long c0 = cyc();
  for (int i = 0; i < NELR; i++) for (int v = 0; v < NVAR; v++) rv[i + v*NELR] = ff_variable[v];
  for (int it = 0; it < ITER; it++) { memcpy(rold, rv, sizeof rv); ref_step_factor(rv, rsf); for (int j = 0; j < RK; j++) { ref_flux(rv, rfl); ref_time_step(j, rold, rv, rsf, rfl); } }
  unsigned long c1 = cyc(); REPORT("cfd scalar", c0, c1, (long)NELR*ITER);
  // hwacha, as euler3d.cpp
  c0 = cyc();
  memset_kernel_ct(NDRANGE1(NELR*NVAR*4, 0), (char *)variables, 0, NELR*NVAR*4);
  initialize_variables_ct(NDRANGE1(NELR, 0), variables, ff_variable, NELR);
  for (int it = 0; it < ITER; it++) {
    memcpy(old_variables, variables, sizeof variables);
    compute_step_factor_ct(NDRANGE1(NELR, 0), variables, areas, step_factors, NELR);
    for (int j = 0; j < RK; j++) {
      compute_flux_ct(NDRANGE1(NELR, 0), ese, normals, variables, ff_variable, fluxes, &ffde, &ffmx, &ffmy, &ffmz, NELR);
      time_step_ct(NDRANGE1(NELR, 0), j, NELR, old_variables, variables, step_factors, fluxes);
    }
  }
  c1 = cyc(); REPORT("cfd hwacha-cc", c0, c1, (long)NELR*ITER);
  int bad = check_f("variables", variables, rv, NELR*NVAR, 1e-3f);
  printf("cfd %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
