# Known issues (hwacha-cc)

## srad: `srad_kernel` indexes its final stores with a clobbered register

`srad_kernel` (NUMBER_THREADS=64, single-precision constants) computes `ei = bx*64 + tx` into vv2
(`vaddw vv2, vs2, vv0`, line 5 of the vf block), reuses vv2 for col / row / float temporaries, and the
three-way phi merge of `d_c_loc` (`if (c < 0) c = 0; else if (c > 1) c = 1;`) is also allocated to vv2
(`@vp6 vaddw vv2, vv8, vs0` / `@vp2 vaddw vv2, vs42, vs0` / `@vp4 vaddw vv2, vs0, vs0`). The stores
`d_dN[ei] = ...` that follow then build their index from vv2 (`@vp1 vsll vv1, vv2, vs49; vsxw ...`),
i.e. from the float bits of `d_c_loc`: STORE ACCESS FAULT on Spike. The register holding `ei` is freed
before its last use by the expander-built store indices. `--no-v32`, `--no-coalesce`, `--no-skip`,
`--no-ct-loops` do not change it. `srad_kernel_vv2_clobber.s` is the assembly. The other five srad
kernels (extract, prepare, reduce, srad2, compress) are not reached.

## backprop: a 16x16 work-group does not fit one stripmine

`bpnn_layerforward_ocl` hard-codes WIDTH = HEIGHT = 16 (a 256-work-item group) and reduces the
products within the group through a barrier-separated tree (`for i in 1,2,4,8,16: weight_matrix[ty][tx]
+= weight_matrix[ty+i/2][tx]; barrier`). hwacha-cc's 2-D flattening maps the group onto 256 lanes,
but with this kernel's register footprint Hwacha grants maxvl = 184 (152 with `--no-v32`), so the
group is executed as two stripmines and the barrier no longer orders the lanes of one group: the
partial sums (and the `w` entries that are written back from them) are wrong, `oldw` (no barrier)
matches. `hwacha_vl_short` reports it. The host cannot shrink the group (the kernel's constants are
the group shape); the kernel would need fewer live registers or hwacha-cc a smaller vector
configuration for this kernel. `backprop_group256_split.s` is the assembly; the host is in `backprop/`.

## myocyte: the kernel does not fit the shared-register file

`kernel_gpu_opencl` is a dispatcher: work-item 0 of group 0 calls `kernel_ecc`, work-item 0 of group 1
calls `kernel_cam` three times; both callees are ~1000-line straight-line ODE right-hand sides. clang
keeps them as calls (too large to inline) and hwacha-cc emits only `__kernel` functions, so the
kernel body is empty and `d_finavalu` stays 0 (60 cycles, 53 mismatches). With the callees marked
`always_inline` (`myocyte.cl` in this directory carries the attribute) hwacha-cc fails with "out of
Hwacha shared registers": the inlined body needs far more than the 64 vs registers for its constants
and uniform temporaries. The host (`myocyte_main.c`, reference = the same .cl compiled as C for the
scalar core, `myocyte_ref.c`) is kept for when hwacha-cc can spill uniform values.

## heartwall: out of shared registers

`kernel_gpu_opencl` (heartwall's single 2235-line kernel: per-frame template matching of 51 sample
points with __local buffers, barriers and a 3-way branch on the point index) needs more than the 64
vs registers Hwacha has for its uniform constants and temporaries: hwacha-cc stops with "out of
Hwacha shared registers". No host was written; `heartwall_kernel_gpu_opencl.ll` is clang's IR of the
unmodified kernel. Needs vs spilling in hwacha-cc.

## dwt2d: out of shared registers

`cl_fdwt53Kernel` (5/3 lifting DWT over a sliding window held in a `__local struct FDWT53` with
per-lane loader / writer structs) compiles with hwacha-cc as of ea483c6^ but, once its struct copies
are expanded into element loads and stores (expandMemIntrinsics, which lavaMD needs), exceeds the
64 vs registers: "out of Hwacha shared registers in cl_fdwt53Kernel". `dwt2d/` keeps the kernel and
a host with a textbook 5/3 reference (not verified against the kernel).
