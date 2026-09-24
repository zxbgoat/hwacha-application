// Shared by every PolyBench host: bare-metal helpers, the hwacha-cc launch conventions.
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "util.h"
static unsigned rng_state = 20240911u;
static inline unsigned rnd(void) { rng_state = rng_state * 1103515245u + 12345u; return rng_state >> 8; }
static inline float frand(float lo, float hi) { return lo + (hi - lo) * (float)(rnd() & 0xffff) / 65535.0f; }
static inline float absf_(float a) { return a < 0 ? -a : a; }
// PolyBench's own error measure (utilities/polybenchUtilFuncts.h): percent difference, with a 0.05%
// threshold in the original hosts. Kept so a case can report exactly the number upstream prints.
static inline float percentDiff(float val1, float val2) {
  if ((absf_(val1) < 0.01f) && (absf_(val2) < 0.01f)) return 0.0f;
  return 100.0f * (absf_(absf_(val1 - val2) / absf_(val1 + 0.00000001f)));
}
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05f
static inline unsigned long cyc(void) { unsigned long c; asm volatile("rdcycle %0" : "=r"(c)); return c; }
#define REPORT(tag, c0, c1, n) printf("%s: %lu cycles, %lu.%02lu cyc/elem\n", tag, (c1)-(c0), ((c1)-(c0))/(n), (((c1)-(c0))*100/(n))%100)
// newlib libm wrappers (exp, pow, ...) set errno through __errno; the bare-metal env has none
static int hw_errno; int *__errno(void) { return &hw_errno; }
// report scalar traps (the crt's default handler only exits with 1337)
#include <stdint.h>
#include <stdlib.h>
static int in_trap;
uintptr_t handle_trap(uintptr_t cause, uintptr_t epc, uintptr_t regs[32]) {
  uintptr_t tval; asm volatile("csrr %0, mtval" : "=r"(tval));
  if (!in_trap++) printf("TRAP cause %lx epc %lx tval %lx\n", cause, epc, tval);   // a nested trap (printf faulted) exits quietly
  exit(1);
}
// hwacha-cc launch conventions: <kernel>_ct(n, args...) runs n work-items in groups of hwacha_group_size
// (0 = one group per stripmine); hwacha_vl_short is set if the hardware could not give a group its size.
extern long hwacha_group_size, hwacha_vl_short;
// 2-D NDRanges (hwacha-cc flattenNDRange): a LS0 x LS1 work-group is one group of LS0*LS1 lanes, the
// NG0 x NG1 groups run sequentially; the kernel derives get_*_id(0/1) from these globals.
long hwacha_ls0, hwacha_ls1, hwacha_ng0, hwacha_ng1;
static long hwacha_last_group;   // the group size the last launch asked for (0 = hardware's choice: no VL-short check)
#define NDRANGE1(n, ls)  (hwacha_last_group = hwacha_group_size = (ls), hwacha_vl_short = 0, (long)(n))
#define NDRANGE2(ng0, ng1, ls0, ls1) (hwacha_ls0 = (ls0), hwacha_ls1 = (ls1), hwacha_ng0 = (ng0), hwacha_ng1 = (ng1), \
                                      hwacha_last_group = hwacha_group_size = (long)(ls0) * (ls1), hwacha_vl_short = 0, (long)(ng0) * (ng1) * (ls0) * (ls1))
#define CEILDIV(a, b) (((a) + (b) - 1) / (b))

// IEEE half precision on the host (gcc 9.2 for riscv64 has no _Float16): the bit pattern in a uint16_t,
// conversions with round-to-nearest-even, as Hwacha's vfcvt.h.s / vfcvt.s.h (Spike cvt16.cc). A half
// arithmetic op on Hwacha is the single-precision op followed by this rounding (vfadd.h etc.), so
// f16(f32 op) reproduces it exactly.
typedef uint16_t f16;
static inline f16 f32_to_f16(float f) {
  uint32_t x; memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u; int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15; uint32_t mant = x & 0x7fffffu;
  if (((x >> 23) & 0xff) == 0xff) return (f16)(sign | 0x7c00u | (mant ? 0x200u : 0));   // inf / nan
  if (exp >= 0x1f) return (f16)(sign | 0x7c00u);                                          // overflow -> inf
  if (exp <= 0) {                                                                          // subnormal / zero
    if (exp < -10) return (f16)sign;
    mant |= 0x800000u; uint32_t shift = (uint32_t)(14 - exp);
    uint32_t h = mant >> shift, rem = mant & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (h & 1))) h++;
    return (f16)(sign | h);
  }
  uint32_t h = sign | ((uint32_t)exp << 10) | (mant >> 13), rem = mant & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) h++;   // may carry into the exponent: correct
  return (f16)h;
}
static inline float f16_to_f32(f16 h) {
  uint32_t sign = ((uint32_t)h & 0x8000u) << 16, exp = (h >> 10) & 0x1f, mant = h & 0x3ffu, x;
  if (exp == 0x1f) x = sign | 0x7f800000u | (mant << 13);
  else if (exp == 0) { if (mant == 0) x = sign; else { int e = -1; do { e++; mant <<= 1; } while (!(mant & 0x400u)); x = sign | ((uint32_t)(127 - 15 - e) << 23) | ((mant & 0x3ffu) << 13); } }
  else x = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  float f; memcpy(&f, &x, 4); return f;
}
static inline float h_round(float f) { return f16_to_f32(f32_to_f16(f)); }   // a half arithmetic result

static int check_f(const char *tag, const float *hw, const float *ref, long n, float rtol) {
  float mx = 0; for (long i = 0; i < n; i++) if (fabsf(ref[i]) > mx) mx = fabsf(ref[i]);
  int bad = 0; float md = 0;
  for (long i = 0; i < n; i++) { float d = fabsf(hw[i] - ref[i]); if (d > md) md = d; if (d > rtol * mx + 1e-6f) bad++; }
  // the bare-metal printf has no %g: micro-units
  printf("%s: %ld values, max|diff|=%ld/1e3 max|ref|=%ld/1e3, %d mismatches%s\n", tag, n, (long)((double)md * 1e3), (long)((double)mx * 1e3), bad, (hwacha_vl_short && hwacha_last_group) ? " (VL SHORT: group size not honoured)" : "");
  return bad;
}
// the same measure the PolyBench hosts print: the number of elements above the percent threshold
static int check_percent(const char *tag, const float *hw, const float *ref, long n) {
  int bad = 0; for (long i = 0; i < n; i++) if (percentDiff(hw[i], ref[i]) > PERCENT_DIFF_ERROR_THRESHOLD) bad++;
  printf("%s: %ld values, %d above %g%%%s\n", tag, n, bad, (double)PERCENT_DIFF_ERROR_THRESHOLD, (hwacha_vl_short && hwacha_last_group) ? " (VL SHORT: group size not honoured)" : "");
  return bad;
}
