// Rodinia lavaMD: N-body forces between particles of neighbouring boxes (kernel_gpu_opencl, one
// work-group of NUMBER_THREADS per box, particles staged through __local); a scalar port of the kernel
// is the reference. boxes1d=2 -> 8 boxes x 100 particles
#include "common.h"
#define B1D 2
#define NBOX (B1D*B1D*B1D)
#define PPB 100
#define NT 64            // NUMBER_THREADS (see Makefile CLDEFS_lavamd)
typedef struct { float v, x, y, z; } FOUR_VECTOR;
typedef struct { int x, y, z, number; long offset; } nei_str;
typedef struct { int x, y, z, number; long offset; int nn; nei_str nei[26]; } box_str;
typedef struct { float alpha; } par_str;
typedef struct { int cur_arg, arch_arg, cores_arg, boxes1d_arg; long number_boxes, box_mem, space_elem, space_mem, space_mem2; } dim_str;
void kernel_gpu_opencl_ct(long n, par_str *par, dim_str *dim, box_str *box, FOUR_VECTOR *rv, float *qv, FOUR_VECTOR *fv);   // struct args are byval (a pointer) in clang's IR
static box_str box[NBOX]; static FOUR_VECTOR rv[NBOX*PPB], fv[NBOX*PPB], rfv[NBOX*PPB]; static float qv[NBOX*PPB];
#define DOT(A, B) ((A).x*(B).x + (A).y*(B).y + (A).z*(B).z)
int main(void) {
  par_str par = {0.5f}; dim_str dim; memset(&dim, 0, sizeof dim); dim.boxes1d_arg = B1D; dim.number_boxes = NBOX; dim.space_elem = NBOX*PPB;
  int nh = 0;   // boxes and their neighbour lists, as main.c
  for (int i = 0; i < B1D; i++) for (int j = 0; j < B1D; j++) for (int k = 0; k < B1D; k++) {
    box[nh].x = k; box[nh].y = j; box[nh].z = i; box[nh].number = nh; box[nh].offset = nh * PPB; box[nh].nn = 0;
    for (int l = -1; l < 2; l++) for (int m = -1; m < 2; m++) for (int n = -1; n < 2; n++) {
      if ((i+l) >= 0 && (j+m) >= 0 && (k+n) >= 0 && (i+l) < B1D && (j+m) < B1D && (k+n) < B1D && !(l == 0 && m == 0 && n == 0)) {
        nei_str *ne = &box[nh].nei[box[nh].nn]; ne->x = k+n; ne->y = j+m; ne->z = i+l; ne->number = ne->z*B1D*B1D + ne->y*B1D + ne->x; ne->offset = ne->number * PPB; box[nh].nn++; } }
    nh++; }
  for (int i = 0; i < NBOX*PPB; i++) { rv[i].v = frand(0.1f, 1); rv[i].x = frand(0.1f, 1); rv[i].y = frand(0.1f, 1); rv[i].z = frand(0.1f, 1); qv[i] = frand(0.1f, 1); fv[i].v = fv[i].x = fv[i].y = fv[i].z = 0; rfv[i] = fv[i]; }
  unsigned long c0 = cyc();
  float a2 = 2 * par.alpha * par.alpha;
  for (int bx = 0; bx < NBOX; bx++) { int fi = box[bx].offset;
    for (int k = 0; k < 1 + box[bx].nn; k++) { int p = k == 0 ? bx : box[bx].nei[k-1].number; int fj = box[p].offset;
      for (int w = 0; w < PPB; w++) for (int j = 0; j < PPB; j++) {
        float r2 = rv[fi+w].v + rv[fj+j].v - DOT(rv[fi+w], rv[fj+j]); float u2 = a2 * r2; float vij = expf(-u2); float fs = 2 * vij;
        float dx = rv[fi+w].x - rv[fj+j].x, dy = rv[fi+w].y - rv[fj+j].y, dz = rv[fi+w].z - rv[fj+j].z;
        rfv[fi+w].v += qv[fj+j]*vij; rfv[fi+w].x += qv[fj+j]*fs*dx; rfv[fi+w].y += qv[fj+j]*fs*dy; rfv[fi+w].z += qv[fj+j]*fs*dz; } } }
  unsigned long c1 = cyc(); REPORT("lavamd scalar", c0, c1, (long)NBOX*PPB);
  c0 = cyc(); kernel_gpu_opencl_ct(NDRANGE1(NBOX*NT, NT), &par, &dim, box, rv, qv, fv); c1 = cyc(); REPORT("lavamd hwacha-cc", c0, c1, (long)NBOX*PPB);
  int bad = check_f("fv", (float *)fv, (float *)rfv, NBOX*PPB*4, 1e-3f);
  printf("lavamd %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
