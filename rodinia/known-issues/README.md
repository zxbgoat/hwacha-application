# Known issues (hwacha-cc) -- all resolved

Every Rodinia case now passes on Spike. This file keeps the diagnosis of the five cases that failed
in the first round and what changed in hwacha-cc for each (hwacha-compiler commit after 3969131).

## srad: an aliased register was freed before its last use (fixed)

`srad_kernel` computed `ei = bx*64 + tx`, and `sext ei` shared its register (no-op casts are
aliases). The alias liveness was extended only at emission time, after `releaseAt` had already freed
the source at its own last use, so the register went to a phi and the final stores indexed with float
bits: STORE ACCESS FAULT. `computePositions` now extends the source's last use to that of every
no-op cast / freeze of it (3969131). A second alias bug surfaced on dwt2d: a no-op cast of a value
that is *not* register-resident (a pooled constant or argument, a spilled value, a rematerialised
address) captured the temporary of one use; casts of such values are now forwarded to the source
and re-derived at each use.

## myocyte / heartwall / dwt2d: out of shared (vs) registers (fixed)

The 64 vs registers held every uniform constant, argument and temporary for the whole kernel.
myocyte (ECC + CaM right-hand sides inlined, ~1000 lines each), heartwall (33 arguments plus the
params struct) and dwt2d (55 constant offsets into one `__local` struct) all exceeded them. Now:

* a per-kernel **uniform pool** (`<kernel>_cpool`, i64 payloads written by the control thread) holds
  numeric constants and, when there are many, the arguments; each use loads into a short-lived
  temporary (`vaddi t, vsPool, 8*i; vlsd t, t`) once fewer than 32 vs registers are free;
* constant addresses into a global (`gep @g, k`) are rematerialised as `vaddi t, base, k`;
* the remaining uniform temporaries **spill**: the live value with the farthest last use is stored
  to `<kernel>_spill` right after its definition (write-through, so the slot is valid wherever the
  register was) and reloaded at each later use.

myocyte additionally needed its callees inlined (`inlineCallees`: clang keeps the two large
right-hand sides as calls, and hwacha-cc emits only `__kernel` functions), `_Z3powff`, `log10f`
and `fmodf` expansions, and per-lane private buffers for `alloca`s (`expandAllocas`).

## dwt2d / heartwall: out of predicate (vp) registers (fixed)

With every callee inlined, `cl_fdwt53Kernel` and heartwall's `kernel_gpu_opencl` keep more than
the 16 predicate registers live: edge masks waiting for their successor blocks, loop active / exit
masks, i1 values. Live i1 values and pending edge masks now **spill to a 32-bit vector register**
(0/1 per lane: `vaddw t, vs0, vs0; @vp vaddw t, vs0, vsOne`) and come back with `vcmpeq` + `vpop`
(not) when consumed. The spill code is hoisted to the top of the block (before its consensual skip
jump) so blocks without active lanes are still skipped -- heartwall faulted on a uniform load in a
block that ran with no active lane because the address had been computed in a skipped block. Two
leaks were fixed on the way: a real header -> exit edge was overwritten by the synthetic loop-exit
edge (its mask never freed), and dead `RegOf` entries were considered live by the spiller.

## backprop / dwt2d: the work-group does not fit one stripmine (fixed)

Spike grants `maxvl = 8 * (256 / (nvv + nvw))` lanes; backprop's 16x16 groups need 256 lanes (at
most 8 vector registers) and dwt2d's 32-lane windows 64. Barrier-separated group algorithms break
when a group is split across stripmines (`hwacha_vl_short`). hwacha-cc now takes a **vector
register cap** (`-vregs N`, or `2048 / reqd_work_group_size` when the kernel declares one) and
spills the live vector values beyond it to per-lane slots of `<kernel>_vspill` (write-through
`vsxd/vsxw` at the definition, reload at each use; header phis of loops inside the vf block are
stored at the latch). Under the cap a narrow value may take a free 64-bit register (every vw is
renamed to a vv anyway). `Makefile` passes `-vregs 8` for backprop, `-vregs 64` for dwt2d and
`-vregs 32` for heartwall (64 lanes per point).

## dwt2d: the reference was wrong (fixed)

The first host compared against a textbook 5/3 transform (rows first, `floorf`, half/half band
layout). The kernel transforms columns first, uses C integer division, and writes four quadrant
bands (`initialize_BandIO`); `dwt2d_main.c` now mirrors that.

## Testing the spillers

`-vpregs N` / `-vsregs N` / `-vregs N` restrict the predicate / shared / vector register files so
the spill paths run on small kernels: `hwacha-cc/test/run` passes with `-vregs 6 -vpregs 8` (pfmin
needs more than 6 vector registers for a 3-operand fmuladd plus addresses). `HWCC_DEBUG_VS=1`
prints what occupies each register when a spiller finds nothing to evict; `HWCC_ANNOTATE=1` writes
the IR instruction above its assembly; `HWCC_TRACE=1` writes the instruction position to a vs
register before every instruction (locate a faulting instruction in Spike's `H:` log);
`HWCC_DUMP_IR=1` prints the kernel IR after preparation.
