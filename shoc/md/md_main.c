// SHOC MD (level1): Lennard-Jones forces with the SHOC compute_lj_force kernel (one work-item per
// atom, neighbour list stored transposed: neighList[j*inum + idx]). As MD.cpp: positions uniform in a
// cube of edge 20, the neighbour list holds each atom's maxNeighbors nearest atoms (built by
// distance sort), cutsq 16, lj1 1.5, lj2 2.0; the check is SHOC's (per-atom relative error of the
// three components below EPSILON = 0.1). nAtom scaled from 12288 to 512, maxNeighbors kept.
#include "common.h"
#define NATOM 512
#define MAXNEIGH 128
#define LS 64
void compute_lj_force_ct(long n, float *force, float *position, int neighCount, int *neighList, float cutsq, float lj1, float lj2, int inum);
static float pos[NATOM*4], force[NATOM*4]; static int neigh[MAXNEIGH*NATOM];
static float dist2(int i, int j) { float dx = pos[i*4] - pos[j*4], dy = pos[i*4+1] - pos[j*4+1], dz = pos[i*4+2] - pos[j*4+2]; return dx*dx + dy*dy + dz*dz; }
static int build_neighbors(void) {   // SHOC insertInOrder / populateNeighborList: the maxNeighbors nearest, ascending
  static int list[MAXNEIGH]; static float d[MAXNEIGH]; int pairs = 0;
  for (int i = 0; i < NATOM; i++) {
    for (int k = 0; k < MAXNEIGH; k++) { list[k] = -1; d[k] = 3.4e38f; }
    for (int j = 0; j < NATOM; j++) { if (i == j) continue; float dij = dist2(i, j); if (dij >= d[MAXNEIGH-1]) continue;
      int k = MAXNEIGH - 1; while (k > 0 && d[k-1] > dij) { d[k] = d[k-1]; list[k] = list[k-1]; k--; } d[k] = dij; list[k] = j; }
    for (int k = 0; k < MAXNEIGH; k++) { neigh[k*NATOM + i] = list[k]; if (d[k] < 16.0f) pairs++; }
  }
  return pairs;
}
int main(void) {
  const float cutsq = 16.0f, lj1 = 1.5f, lj2 = 2.0f;
  for (int i = 0; i < NATOM; i++) { pos[i*4] = frand(0, 20); pos[i*4+1] = frand(0, 20); pos[i*4+2] = frand(0, 20); pos[i*4+3] = 0; }
  int pairs = build_neighbors();
  printf("md: %d of %d pairs within the cutoff\n", pairs, NATOM*MAXNEIGH);
  unsigned long c0 = cyc(); compute_lj_force_ct(NDRANGE1(NATOM, LS), force, pos, MAXNEIGH, neigh, cutsq, lj1, lj2, NATOM); unsigned long c1 = cyc();
  REPORT("md hwacha-cc", c0, c1, (long)NATOM*MAXNEIGH);
  int bad = 0; c0 = cyc();
  for (int i = 0; i < NATOM; i++) {
    float fx = 0, fy = 0, fz = 0;
    for (int j = 0; j < MAXNEIGH; j++) { int jidx = neigh[j*NATOM + i]; float delx = pos[i*4] - pos[jidx*4], dely = pos[i*4+1] - pos[jidx*4+1], delz = pos[i*4+2] - pos[jidx*4+2];
      float r2inv = delx*delx + dely*dely + delz*delz; if (r2inv < cutsq) { r2inv = 1.0f/r2inv; float r6inv = r2inv*r2inv*r2inv; float fc = r2inv*r6inv*(lj1*r6inv - lj2); fx += delx*fc; fy += dely*fc; fz += delz*fc; } }
    float ex = (force[i*4] - fx) / force[i*4], ey = (force[i*4+1] - fy) / force[i*4+1], ez = (force[i*4+2] - fz) / force[i*4+2];
    float err = sqrtf(ex*ex) + sqrtf(ey*ey) + sqrtf(ez*ez);
    if (err > 0.1f) { if (bad < 4) printf("  atom %d: err %ld/1e6\n", i, (long)(err*1e6f)); bad++; }
  }
  c1 = cyc(); REPORT("md scalar", c0, c1, (long)NATOM*MAXNEIGH);
  printf("md: %d atoms above EPSILON / %d\n", bad, NATOM);
  printf("md %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
