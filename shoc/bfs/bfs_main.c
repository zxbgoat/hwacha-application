// SHOC BFS (level1): level-synchronous breadth-first search with the SHOC BFS_kernel_warp
// (bfs_iiit.cl: each warp of W_SZ work-items scans CHUNK_SZ vertices, the warp's lanes expand the
// neighbours of a vertex at the current level and set the level of unvisited ones; a flag tells the
// host whether another level exists), launched as BFS.cpp RunTest1: one launch per level until the
// flag stays clear. Graph: Graph::GenerateSimpleKWayGraph (vertex i -> i*degree+1 .. i*degree+degree
// and back to (i-1)/degree), degree 2 as SHOC's default, source vertex 0; the reference is a scalar
// BFS and the check compares every vertex's level (SHOC verify_results). bfs_uiuc_spill.cl (SHOC's
// second variant, atomics + __local queues) is compiled into bfs.s but its host is not written.
#include "common.h"
#define NV 2048
#define DEGREE 2
#define W_SZ 32
#define CHUNK_SZ 32
#define LS 64
void BFS_kernel_warp_ct(long n, unsigned *levels, unsigned *edgeArray, unsigned *edgeArrayAux, int W_SZ_, int CHUNK_SZ_, unsigned numVertices, int curr, int *flag);
static unsigned edgeOff[NV+1], edgeList[NV*(DEGREE+1)], levels[NV], ref[NV], queue_[NV];
int main(void) {
  unsigned off = 0;
  for (unsigned i = 0; i < NV; i++) { edgeOff[i] = off; for (unsigned j = 0; j < DEGREE; j++) { unsigned t = i*DEGREE + j + 1; if (t < NV) edgeList[off++] = t; } if (i != 0) edgeList[off++] = (i-1) / DEGREE; }
  edgeOff[NV] = off;
  for (int i = 0; i < NV; i++) { levels[i] = 0xffffffffu; ref[i] = 0xffffffffu; }
  levels[0] = 0; ref[0] = 0;
  unsigned long c0 = cyc();
  { int qh = 0, qt = 0; queue_[qt++] = 0; while (qh < qt) { unsigned v = queue_[qh++]; for (unsigned e = edgeOff[v]; e < edgeOff[v+1]; e++) { unsigned w = edgeList[e]; if (ref[w] == 0xffffffffu) { ref[w] = ref[v] + 1; queue_[qt++] = w; } } } }
  unsigned long c1 = cyc(); REPORT("bfs scalar", c0, c1, NV);
  int flag = 1, iters = 0; c0 = cyc();
  while (flag) { flag = 0; BFS_kernel_warp_ct(NDRANGE1(CEILDIV(NV, LS)*LS, LS), levels, edgeOff, edgeList, W_SZ, CHUNK_SZ, NV, iters, &flag); iters++; }
  c1 = cyc(); REPORT("bfs hwacha-cc", c0, c1, NV);
  int bad = 0; unsigned maxl = 0; for (int i = 0; i < NV; i++) { if (levels[i] != ref[i]) bad++; if (ref[i] != 0xffffffffu && ref[i] > maxl) maxl = ref[i]; }
  printf("bfs: %d level mismatches / %d vertices, %d levels, %d launches\n", bad, NV, maxl + 1, iters);
  printf("bfs %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
