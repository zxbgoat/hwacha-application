// Rodinia heartwall (OpenCL): kernel_gpu_opencl tracks 51 points (20 endocardium + 31 epicardium) of
// a heart ultrasound video, one work-group (NUMBER_THREADS = RD_WG_SIZE = 64 lanes) per point. Frame 0
// extracts the template around each point; every later frame correlates the template over a
// (2 sSize + 1)^2 search window (normalised cross-correlation via cumulative sums), applies the
// displacement mask and writes the tracked row / column. The reference is the Rodinia OpenMP kernel
// (heartwall_ref.c) on the same synthetic frames (a pseudo-random texture shifted by one row and one
// column per frame); the tracked locations and the templates must match the reference exactly.
// tSize / sSize are scaled down from 25 / 40 to keep the Spike run short; the parameter derivation is
// the original (main.c of the OpenCL and OpenMP versions).
#include "common.h"
#define RD_WG_SIZE 64   // NUMBER_THREADS: must match CLDEFS_heartwall in the Makefile
#include "main.h"
#include "heartwall_ref.h"

#define FRAMES 3
#define ROWS 560
#define COLS 480
#define T_SIZE 5
#define S_SIZE 8
#define MAX_MOVE 10
#define NT NUMBER_THREADS

void kernel_gpu_opencl_ct(long n, params_common d_common, fp *d_frame, int d_frame_no,
  int *d_endoRow, int *d_endoCol, int *d_tEndoRowLoc, int *d_tEndoColLoc,
  int *d_epiRow, int *d_epiCol, int *d_tEpiRowLoc, int *d_tEpiColLoc,
  fp *d_endoT, fp *d_epiT, fp *d_in2_all, fp *d_conv_all, fp *d_in2_pad_cumv_all, fp *d_in2_pad_cumv_sel_all,
  fp *d_in2_sub_cumh_all, fp *d_in2_sub_cumh_sel_all, fp *d_in2_sub2_all, fp *d_in2_sqr_all, fp *d_in2_sqr_sub2_all,
  fp *d_in_sqr_all, fp *d_tMask_all, fp *d_mask_conv_all, fp *d_in_mod_temp_all, fp *in_partial_sum_all,
  fp *in_sqr_partial_sum_all, fp *par_max_val_all, int *par_max_coo_all, fp *in_final_sum_all,
  fp *in_sqr_final_sum_all, fp *denomT_all, fp *checksum);

static int endoRow[ENDO_POINTS] = {369,400,429,452,476,486,479,458,433,404,374,346,318,294,277,269,275,287,311,339};
static int endoCol[ENDO_POINTS] = {408,406,397,383,354,322,294,270,250,237,235,241,254,273,300,328,356,383,401,411};
static int epiRow[EPI_POINTS] = {390,419,448,474,501,519,535,542,543,538,528,511,491,466,438,406,376,347,318,291,275,259,256,252,252,257,266,283,305,331,360};
static int epiCol[EPI_POINTS] = {457,454,446,431,411,388,361,331,301,273,243,218,196,178,166,157,155,165,177,197,218,248,276,304,333,361,391,415,434,448,455};

static fp frames[FRAMES][ROWS * COLS];   // column-major, as the kernels index it: frame[col * rows + row]
static char pool[12 << 20]; static long pool_top;
static void *xalloc(long bytes) { void *p = pool + pool_top; pool_top += (bytes + 15) & ~15L; if (pool_top > (long)sizeof pool) { printf("pool exhausted\n"); exit(2); } return p; }
static unsigned tex(int r, int c) { unsigned h = (unsigned)r * 1103515245u + (unsigned)c * 12345u + (unsigned)(r * 31 + c) * 2654435761u; h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15; return h & 0xff; }

int main(void) {
  for (int f = 0; f < FRAMES; f++) for (int c = 0; c < COLS; c++) for (int r = 0; r < ROWS; r++) frames[f][c * ROWS + r] = (fp)tex(r - f, c - f);

  // ---- OpenCL side: params_common exactly as the Rodinia OpenCL main.c derives it
  params_common common; memset(&common, 0, sizeof common);
  common.frames_processed = FRAMES; common.no_frames = FRAMES; common.frame_rows = ROWS; common.frame_cols = COLS;
  common.frame_elem = ROWS * COLS; common.frame_mem = sizeof(fp) * common.frame_elem;
  common.tSize = T_SIZE; common.sSize = S_SIZE; common.maxMove = MAX_MOVE; common.alpha = 0.87f;
  common.endoPoints = ENDO_POINTS; common.epiPoints = EPI_POINTS; common.allPoints = ALL_POINTS;
  common.endo_mem = sizeof(int) * common.endoPoints; common.epi_mem = sizeof(int) * common.epiPoints;
  common.in_rows = common.tSize + 1 + common.tSize;
  common.in_cols = common.in_rows;
  common.in_elem = common.in_rows * common.in_cols;
  common.in2_rows = common.sSize + 1 + common.sSize;
  common.in2_cols = common.in2_rows;
  common.in2_elem = common.in2_rows * common.in2_cols;
  common.conv_rows = common.in_rows + common.in2_rows - 1;												// number of rows in I
  common.conv_cols = common.in_cols + common.in2_cols - 1;												// number of columns in I
  common.conv_elem = common.conv_rows * common.conv_cols;													// number of elements
  common.ioffset = 0;
  common.joffset = 0;
  common.in2_pad_add_rows = common.in_rows;
  common.in2_pad_add_cols = common.in_cols;
  common.in2_pad_cumv_rows = common.in2_rows + 2*common.in2_pad_add_rows;
  common.in2_pad_cumv_cols = common.in2_cols + 2*common.in2_pad_add_cols;
  common.in2_pad_cumv_elem = common.in2_pad_cumv_rows * common.in2_pad_cumv_cols;
  common.in2_pad_cumv_sel_rowlow = 1 + common.in_rows;													// (1 to n+1)
  common.in2_pad_cumv_sel_rowhig = common.in2_pad_cumv_rows - 1;
  common.in2_pad_cumv_sel_collow = 1;
  common.in2_pad_cumv_sel_colhig = common.in2_pad_cumv_cols;
  common.in2_pad_cumv_sel_elem = common.in2_pad_cumv_sel_rows * common.in2_pad_cumv_sel_cols;
  common.in2_pad_cumv_sel2_rowlow = 1;
  common.in2_pad_cumv_sel2_rowhig = common.in2_pad_cumv_rows - common.in_rows - 1;
  common.in2_pad_cumv_sel2_collow = 1;
  common.in2_pad_cumv_sel2_colhig = common.in2_pad_cumv_cols;
  common.in2_sub_cumh_elem = common.in2_sub_cumh_rows * common.in2_sub_cumh_cols;
  common.in2_sub_cumh_sel_rowlow = 1;
  common.in2_sub_cumh_sel_rowhig = common.in2_sub_cumh_rows;
  common.in2_sub_cumh_sel_collow = 1 + common.in_cols;
  common.in2_sub_cumh_sel_colhig = common.in2_sub_cumh_cols - 1;
  common.in2_sub_cumh_sel_elem = common.in2_sub_cumh_sel_rows * common.in2_sub_cumh_sel_cols;
  common.in2_sub_cumh_sel2_rowlow = 1;
  common.in2_sub_cumh_sel2_rowhig = common.in2_sub_cumh_rows;
  common.in2_sub_cumh_sel2_collow = 1;
  common.in2_sub_cumh_sel2_colhig = common.in2_sub_cumh_cols - common.in_cols - 1;
  common.in2_sub2_elem = common.in2_sub2_rows * common.in2_sub2_cols;
  common.in2_sqr_rows = common.in2_rows;
  common.in2_sqr_cols = common.in2_cols;
  common.in2_sqr_elem = common.in2_elem;
  common.in2_sqr_mem = common.in2_mem;
  common.in2_sqr_sub2_rows = common.in2_sub2_rows;
  common.in2_sqr_sub2_cols = common.in2_sub2_cols;
  common.in2_sqr_sub2_elem = common.in2_sub2_elem;
  common.in2_sqr_sub2_mem = common.in2_sub2_mem;
  common.in_sqr_rows = common.in_rows;
  common.in_sqr_cols = common.in_cols;
  common.in_sqr_elem = common.in_elem;
  common.in_sqr_mem = common.in_mem;
  common.mask_rows = common.maxMove;
  common.mask_cols = common.mask_rows;
  common.mask_elem = common.mask_rows * common.mask_cols;
  common.mask_conv_rows = common.tMask_rows;												// number of rows in I
  common.mask_conv_cols = common.tMask_cols;												// number of columns in I
  common.mask_conv_elem = common.mask_conv_rows * common.mask_conv_cols;												// number of elements
  common.mask_conv_ioffset = (common.mask_rows-1)/2;
  common.mask_conv_joffset = (common.mask_cols-1)/2;
  if((common.mask_rows-1) % 2 > 0.5) common.mask_conv_ioffset = common.mask_conv_ioffset + 1;
  if((common.mask_cols-1) % 2 > 0.5) common.mask_conv_joffset = common.mask_conv_joffset + 1;
  int *tEndoRowLoc = xalloc(common.endo_mem * FRAMES), *tEndoColLoc = xalloc(common.endo_mem * FRAMES);
  int *tEpiRowLoc = xalloc(common.epi_mem * FRAMES), *tEpiColLoc = xalloc(common.epi_mem * FRAMES);
  fp *endoT = xalloc(sizeof(fp) * common.in_elem * ENDO_POINTS), *epiT = xalloc(sizeof(fp) * common.in_elem * EPI_POINTS);
  int A = ALL_POINTS;
  fp *in2_all = xalloc(sizeof(fp) * common.in2_elem * A), *conv_all = xalloc(sizeof(fp) * common.conv_elem * A);
  fp *in2_pad_cumv_all = xalloc(sizeof(fp) * common.in2_pad_cumv_elem * A), *in2_pad_cumv_sel_all = xalloc(sizeof(fp) * common.in2_pad_cumv_sel_elem * A);
  fp *in2_sub_cumh_all = xalloc(sizeof(fp) * common.in2_sub_cumh_elem * A), *in2_sub_cumh_sel_all = xalloc(sizeof(fp) * common.in2_sub_cumh_sel_elem * A);
  fp *in2_sub2_all = xalloc(sizeof(fp) * common.in2_sub2_elem * A), *in2_sqr_all = xalloc(sizeof(fp) * common.in2_sqr_elem * A);
  fp *in2_sqr_sub2_all = xalloc(sizeof(fp) * common.in2_sqr_sub2_elem * A), *in_sqr_all = xalloc(sizeof(fp) * common.in_sqr_elem * A);
  fp *tMask_all = xalloc(sizeof(fp) * common.tMask_elem * A), *mask_conv_all = xalloc(sizeof(fp) * common.mask_conv_elem * A);
  fp *in_mod_temp_all = xalloc(sizeof(fp) * common.in_elem * A), *in_partial_sum_all = xalloc(sizeof(fp) * common.in_cols * A);
  fp *in_sqr_partial_sum_all = xalloc(sizeof(fp) * common.in_sqr_rows * A), *par_max_val_all = xalloc(sizeof(fp) * common.mask_conv_rows * A);
  int *par_max_coo_all = xalloc(sizeof(int) * common.mask_conv_rows * A);
  fp *in_final_sum_all = xalloc(sizeof(fp) * A), *in_sqr_final_sum_all = xalloc(sizeof(fp) * A), *denomT_all = xalloc(sizeof(fp) * A), *checksum = xalloc(sizeof(fp) * 100);

  unsigned long c0 = cyc();
  for (int f = 0; f < FRAMES; f++)
    kernel_gpu_opencl_ct(NDRANGE1((long)A * NT, NT), common, frames[f], f, endoRow, endoCol, tEndoRowLoc, tEndoColLoc, epiRow, epiCol, tEpiRowLoc, tEpiColLoc,
      endoT, epiT, in2_all, conv_all, in2_pad_cumv_all, in2_pad_cumv_sel_all, in2_sub_cumh_all, in2_sub_cumh_sel_all, in2_sub2_all, in2_sqr_all, in2_sqr_sub2_all,
      in_sqr_all, tMask_all, mask_conv_all, in_mod_temp_all, in_partial_sum_all, in_sqr_partial_sum_all, par_max_val_all, par_max_coo_all, in_final_sum_all, in_sqr_final_sum_all, denomT_all, checksum);
  unsigned long c1 = cyc(); REPORT("heartwall hwacha-cc", c0, c1, (long)A * FRAMES);

  // ---- reference: the OpenMP kernel with its own public_struct / private_struct (main.c of the OpenMP version)
  public_struct public; memset(&public, 0, sizeof public); private_struct private[ALL_POINTS];
  public.endoPoints = ENDO_POINTS; public.epiPoints = EPI_POINTS; public.allPoints = ALL_POINTS; public.frames = FRAMES;
  public.frame_rows = ROWS; public.frame_cols = COLS; public.frame_elem = ROWS * COLS;
  public.tSize = T_SIZE; public.sSize = S_SIZE; public.maxMove = MAX_MOVE; public.alpha = 0.87f;
  public.d_endoRow = endoRow; public.d_endoCol = endoCol; public.d_epiRow = epiRow; public.d_epiCol = epiCol;
  public.d_tEndoRowLoc = xalloc(common.endo_mem * FRAMES); public.d_tEndoColLoc = xalloc(common.endo_mem * FRAMES);
  public.d_tEpiRowLoc = xalloc(common.epi_mem * FRAMES); public.d_tEpiColLoc = xalloc(common.epi_mem * FRAMES);
  public.in2_elem = public.in2_rows * public.in2_cols;
  public.in_mod_rows = public.tSize+1+public.tSize;
  public.in_mod_cols = public.in_mod_rows;
  public.in_mod_elem = public.in_mod_rows * public.in_mod_cols;
  public.ioffset = 0;
  public.joffset = 0;
  public.conv_rows = public.in_mod_rows + public.in2_rows - 1;												// number of rows in I
  public.conv_cols = public.in_mod_cols + public.in2_cols - 1;												// number of columns in I
  public.conv_elem = public.conv_rows * public.conv_cols;												// number of elements
  public.in2_pad_add_rows = public.in_mod_rows;
  public.in2_pad_add_cols = public.in_mod_cols;
  public.in2_pad_rows = public.in2_rows + 2*public.in2_pad_add_rows;
  public.in2_pad_cols = public.in2_cols + 2*public.in2_pad_add_cols;
  public.in2_pad_elem = public.in2_pad_rows * public.in2_pad_cols;
  public.in2_pad_cumv_sel_rowlow = 1 + public.in_mod_rows;													// (1 to n+1)
  public.in2_pad_cumv_sel_rowhig = public.in2_pad_rows - 1;
  public.in2_pad_cumv_sel_collow = 1;
  public.in2_pad_cumv_sel_colhig = public.in2_pad_cols;
  public.in2_pad_cumv_sel2_rowlow = 1;
  public.in2_pad_cumv_sel2_rowhig = public.in2_pad_rows - public.in_mod_rows - 1;
  public.in2_pad_cumv_sel2_collow = 1;
  public.in2_pad_cumv_sel2_colhig = public.in2_pad_cols;
  public.in2_sub_elem = public.in2_sub_rows * public.in2_sub_cols;
  public.in2_sub_cumh_sel_rowlow = 1;
  public.in2_sub_cumh_sel_rowhig = public.in2_sub_rows;
  public.in2_sub_cumh_sel_collow = 1 + public.in_mod_cols;
  public.in2_sub_cumh_sel_colhig = public.in2_sub_cols - 1;
  public.in2_sub_cumh_sel2_rowlow = 1;
  public.in2_sub_cumh_sel2_rowhig = public.in2_sub_rows;
  public.in2_sub_cumh_sel2_collow = 1;
  public.in2_sub_cumh_sel2_colhig = public.in2_sub_cols - public.in_mod_cols - 1;
  public.in2_sub2_sqr_elem = public.in2_sub2_sqr_rows * public.in2_sub2_sqr_cols;
  public.mask_rows = public.maxMove;
  public.mask_cols = public.mask_rows;
  public.mask_elem = public.mask_rows * public.mask_cols;
  public.mask_conv_rows = public.tMask_rows;												// number of rows in I
  public.mask_conv_cols = public.tMask_cols;												// number of columns in I
  public.mask_conv_elem = public.mask_conv_rows * public.mask_conv_cols;												// number of elements
  public.mask_conv_ioffset = (public.mask_rows-1)/2;
  public.mask_conv_joffset = (public.mask_cols-1)/2;
  if((public.mask_rows-1) % 2 > 0.5) public.mask_conv_ioffset = public.mask_conv_ioffset + 1;
  if((public.mask_cols-1) % 2 > 0.5) public.mask_conv_joffset = public.mask_conv_joffset + 1;
  public.d_endoT = xalloc(sizeof(fp) * public.in_mod_elem * ENDO_POINTS); public.d_epiT = xalloc(sizeof(fp) * public.in_mod_elem * EPI_POINTS);
  for (int i = 0; i < ALL_POINTS; i++) {
    private_struct *p = &private[i];
    p->in_partial_sum = xalloc(sizeof(fp) * (2 * T_SIZE + 1)); p->in_sqr_partial_sum = xalloc(sizeof(fp) * (2 * T_SIZE + 1));
    p->par_max_val = xalloc(sizeof(fp) * (2 * T_SIZE + 2 * S_SIZE + 1)); p->par_max_coo = xalloc(sizeof(int) * (2 * T_SIZE + 2 * S_SIZE + 1));
    p->d_in2 = xalloc(sizeof(fp) * public.in2_elem); p->d_in2_sqr = xalloc(sizeof(fp) * public.in2_elem);
    p->d_in_mod = xalloc(sizeof(fp) * public.in_mod_elem); p->d_in_sqr = xalloc(sizeof(fp) * public.in_mod_elem);
    p->d_conv = xalloc(sizeof(fp) * public.conv_elem); p->d_in2_pad = xalloc(sizeof(fp) * public.in2_pad_elem);
    p->d_in2_sub = xalloc(sizeof(fp) * public.in2_sub_elem); p->d_in2_sub2_sqr = xalloc(sizeof(fp) * public.in2_sub2_sqr_elem);
    p->d_tMask = xalloc(sizeof(fp) * public.tMask_elem); p->d_mask_conv = xalloc(sizeof(fp) * public.mask_conv_elem);
    if (i < ENDO_POINTS) { p->point_no = i; p->d_Row = endoRow; p->d_Col = endoCol; p->d_tRowLoc = public.d_tEndoRowLoc; p->d_tColLoc = public.d_tEndoColLoc; p->d_T = public.d_endoT; }
    else { p->point_no = i - ENDO_POINTS; p->d_Row = epiRow; p->d_Col = epiCol; p->d_tRowLoc = public.d_tEpiRowLoc; p->d_tColLoc = public.d_tEpiColLoc; p->d_T = public.d_epiT; }
    p->in_pointer = p->point_no * public.in_mod_elem;
  }
  c0 = cyc();
  for (public.frame_no = 0; public.frame_no < FRAMES; public.frame_no++) { public.d_frame = frames[public.frame_no]; for (int i = 0; i < ALL_POINTS; i++) heartwall_ref(public, private[i]); }
  c1 = cyc(); REPORT("heartwall scalar", c0, c1, (long)A * FRAMES);

  // ---- compare: tracked locations of every point in every frame (exact), templates (exact copies of frame 0)
  int bad = 0;
  for (int f = 1; f < FRAMES; f++) {
    for (int i = 0; i < ENDO_POINTS; i++) { int k = i * FRAMES + f; if (tEndoRowLoc[k] != public.d_tEndoRowLoc[k] || tEndoColLoc[k] != public.d_tEndoColLoc[k]) { if (bad < 6) printf("  endo point %d frame %d: hw (%d,%d) ref (%d,%d)\n", i, f, tEndoRowLoc[k], tEndoColLoc[k], public.d_tEndoRowLoc[k], public.d_tEndoColLoc[k]); bad++; } }
    for (int i = 0; i < EPI_POINTS; i++) { int k = i * FRAMES + f; if (tEpiRowLoc[k] != public.d_tEpiRowLoc[k] || tEpiColLoc[k] != public.d_tEpiColLoc[k]) { if (bad < 6) printf("  epi point %d frame %d: hw (%d,%d) ref (%d,%d)\n", i, f, tEpiRowLoc[k], tEpiColLoc[k], public.d_tEpiRowLoc[k], public.d_tEpiColLoc[k]); bad++; } }
  }
  printf("  tracked: endo point 0 (%d,%d) -> frames 1..%d hw (%d,%d) (%d,%d), ref (%d,%d) (%d,%d)\n", endoRow[0], endoCol[0], FRAMES - 1,
         tEndoRowLoc[1], tEndoColLoc[1], tEndoRowLoc[2], tEndoColLoc[2], public.d_tEndoRowLoc[1], public.d_tEndoColLoc[1], public.d_tEndoRowLoc[2], public.d_tEndoColLoc[2]);
  int badT = 0;
  for (int i = 0; i < common.in_elem * ENDO_POINTS; i++) if (endoT[i] != public.d_endoT[i]) badT++;
  for (int i = 0; i < common.in_elem * EPI_POINTS; i++) if (epiT[i] != public.d_epiT[i]) badT++;
  printf("heartwall: %d location mismatches / %d, %d template mismatches / %d%s\n", bad, ALL_POINTS * (FRAMES - 1), badT, common.in_elem * ALL_POINTS, hwacha_vl_short ? " (VL SHORT)" : "");
  printf("heartwall %s\n", (bad || badT) ? "FAIL" : "PASS"); return (bad || badT) != 0;
}
