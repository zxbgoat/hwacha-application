// Rodinia b+tree: findK (point queries) and findRangeK (range queries) over a B+ tree flattened into
// knode arrays, one work-group of ORDER+1 lanes per query (one lane per key slot), driven like the two
// kernel wrappers; a scalar port of the kernels is the reference
#include "common.h"
#include <stdbool.h>
#define ORDER 63
#define NKEYS 512          // records / leaf keys
#define NQ 32
typedef struct { int value; } record;
typedef struct { int location; int indices[ORDER + 1]; int keys[ORDER + 1]; bool is_leaf; int num_keys; } knode;
void findK_ct(long n, long height, knode *knodes, long knodes_elem, record *records, long *currKnode, long *offset, int *keys, record *ans);
void findRangeK_ct(long n, long height, knode *knodes, long knodes_elem, long *currKnode, long *offset, long *lastKnode, long *offset_2, int *start, int *end, int *recstart, int *reclen);
static record records[NKEYS]; static knode knodes[64]; static long cur[NQ], off[NQ], last[NQ], off2[NQ], rcur[NQ], roff[NQ], rlast[NQ], roff2[NQ];
static int keys[NQ], starts[NQ], ends[NQ], recstart[NQ], reclen[NQ], rrecstart[NQ], rreclen[NQ]; static record ans[NQ], rans[NQ];
int main(void) {
  // a 2-level tree over the sorted keys 0..NKEYS-1 (record[k].value = 3k): root has 8 children, each leaf holds 64 keys
  int nleaf = NKEYS / (ORDER + 1), height = 1, nel = 1 + nleaf;
  for (int i = 0; i < NKEYS; i++) records[i].value = 3 * i;
  memset(knodes, 0, sizeof knodes);
  knodes[0].is_leaf = false; knodes[0].num_keys = nleaf;
  for (int c = 0; c < nleaf; c++) { knodes[0].keys[c] = c * (ORDER + 1); knodes[0].indices[c] = 1 + c; }
  knodes[0].keys[nleaf] = NKEYS; for (int c = nleaf; c <= ORDER; c++) knodes[0].indices[c] = nel;   // out-of-range slots: index >= knodes_elem (the kernel's guard)
  for (int c = 0; c < nleaf; c++) { knode *kn = &knodes[1 + c]; kn->is_leaf = true; kn->num_keys = ORDER + 1;
    for (int t = 0; t <= ORDER; t++) { kn->keys[t] = c * (ORDER + 1) + t; kn->indices[t] = c * (ORDER + 1) + t; } }
  for (int q = 0; q < NQ; q++) { keys[q] = rnd() % NKEYS; starts[q] = rnd() % (NKEYS - 8); ends[q] = starts[q] + rnd() % 8; cur[q] = rcur[q] = 0; off[q] = roff[q] = 0; last[q] = rlast[q] = 0; off2[q] = roff2[q] = 0; ans[q].value = rans[q].value = -1; recstart[q] = rrecstart[q] = reclen[q] = rreclen[q] = -1; }
  // reference: the kernels' work-group program, lanes in order, barriers = phase boundaries
  unsigned long c0 = cyc();
  for (int b = 0; b < NQ; b++) {
    for (int i = 0; i < height; i++) { for (int t = 0; t <= ORDER; t++) if (knodes[rcur[b]].keys[t] <= keys[b] && knodes[rcur[b]].keys[t+1] > keys[b] && knodes[roff[b]].indices[t] < nel) roff[b] = knodes[roff[b]].indices[t]; rcur[b] = roff[b]; }
    for (int t = 0; t <= ORDER; t++) if (knodes[rcur[b]].keys[t] == keys[b]) rans[b].value = records[knodes[rcur[b]].indices[t]].value; }
  for (int q = 0; q < NQ; q++) { rcur[q] = 0; roff[q] = 0; rlast[q] = 0; roff2[q] = 0; }
  for (int b = 0; b < NQ; b++) {
    for (int i = 0; i < height; i++) { for (int t = 0; t <= ORDER; t++) { if (knodes[rcur[b]].keys[t] <= starts[b] && knodes[rcur[b]].keys[t+1] > starts[b] && knodes[rcur[b]].indices[t] < nel) roff[b] = knodes[rcur[b]].indices[t];
        if (knodes[rlast[b]].keys[t] <= ends[b] && knodes[rlast[b]].keys[t+1] > ends[b] && knodes[rlast[b]].indices[t] < nel) roff2[b] = knodes[rlast[b]].indices[t]; } rcur[b] = roff[b]; rlast[b] = roff2[b]; }
    for (int t = 0; t <= ORDER; t++) if (knodes[rcur[b]].keys[t] == starts[b]) rrecstart[b] = knodes[rcur[b]].indices[t];
    for (int t = 0; t <= ORDER; t++) if (knodes[rlast[b]].keys[t] == ends[b]) rreclen[b] = knodes[rlast[b]].indices[t] - rrecstart[b] + 1; }
  // (reference kept as the kernel's lane program; the range kernel is launched on its own fresh cur/last)
  unsigned long c1 = cyc(); REPORT("btree scalar", c0, c1, (long)NQ);
  // the range kernel starts from the root again (its own cur/last arrays)
  for (int q = 0; q < NQ; q++) { cur[q] = 0; off[q] = 0; last[q] = 0; off2[q] = 0; }
  c0 = cyc();
  { static long c2[NQ], o2[NQ]; memset(c2, 0, sizeof c2); memset(o2, 0, sizeof o2);
    findK_ct(NDRANGE1(NQ * (ORDER + 1), ORDER + 1), height, knodes, nel, records, c2, o2, keys, ans); }
  findRangeK_ct(NDRANGE1(NQ * (ORDER + 1), ORDER + 1), height, knodes, nel, cur, off, last, off2, starts, ends, recstart, reclen);
  c1 = cyc(); REPORT("btree hwacha-cc", c0, c1, (long)NQ);
  int bad = 0; for (int q = 0; q < NQ; q++) { if (ans[q].value != rans[q].value) bad++; if (recstart[q] != rrecstart[q] || reclen[q] != rreclen[q]) bad++; }
  for (int q = 0; q < 4; q++) printf("  q%d key %d: hw %d ref %d | range [%d,%d]: hw (%d,%d) ref (%d,%d)\n", q, keys[q], ans[q].value, rans[q].value, starts[q], ends[q], recstart[q], reclen[q], rrecstart[q], rreclen[q]);
  printf("btree: %d mismatches / %d queries (point + range)%s\n", bad, NQ, hwacha_vl_short ? " (VL SHORT)" : "");
  printf("btree %s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
