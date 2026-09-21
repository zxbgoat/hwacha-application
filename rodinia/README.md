# Rodinia on Hwacha

The Rodinia 3.1 OpenCL benchmarks, one directory per application: the unmodified Rodinia kernel
(`<app>.cl`, from `rodinia_3.1/opencl/<app>/`), hwacha-cc's assembly of it (`<app>.s`, via
`clang -x cl -emit-llvm` -> `hwacha-cc`), and a bare-metal host (`<app>_main.c` + `common.h`) that
builds a small random input, runs a scalar reference and the Hwacha kernels, compares them and prints
`<app> PASS` with both cycle counts. No PyTorch in the loop: the kernels are the original OpenCL
work-item programs, hwacha-cc maps work-items to lanes, and the host calls the generated
`<kernel>_ct(n, args...)` control-thread entries with the arguments the OpenCL host would pass.
`common.h`'s `NDRANGE1(n, local)` / `NDRANGE2(ng0, ng1, ls0, ls1)` set the work-group shape for a
launch (2-D groups are flattened by hwacha-cc, see its `flattenNDRange`).

`make` builds all, `make run` runs all on Spike, `make <app>` / `make <app>.spike` for one,
`make gen-<app>` recompiles the assembly from the `.cl` (needs `~/miniforge3/bin/clang` and the
hwacha-cc build tree; `CLDEFS_<app>` in the Makefile holds the defines the host would pass).
Cycle counts are Spike's `rdcycle`, i.e. instructions retired by the control thread; the vector
work is not in them (RTL numbers for the first five are in hwacha-compiler's NOTES).

## Cases

| case | Rodinia kernel(s) | problem size | Hwacha |
|---|---|---|---|
| nn | `NearestNeighbor` | 2048 records | PASS |
| kmeans | `kmeans_swap` + `kmeans_kernel_c` | 1024 x 8, 5 clusters | PASS |
| bfs | `BFS_1` + `BFS_2`, level-synchronous from the host | 2048 nodes, degree 4 | PASS |
| pgain | streamcluster `memset_kernel` + `pgain_kernel` | 1024 x 8, 4 centers | PASS |
| pathfinder | `dynproc_kernel` | 8 x 1024, pyramid 2 | PASS |
| gaussian | `Fan1` + `Fan2` (2-D) per pivot | 64 x 64 system | PASS |
| hotspot | `hotspot` (2-D, 8x8 tiles, pyramid 2) | 64 x 64, 4 steps | PASS |
| hotspot3d | `hotspotOpt1` (2-D, z sweep) | 32 x 32 x 8, 2 steps | PASS |
| lud | `lud_diagonal` + `lud_perimeter` + `lud_internal` (2-D) | 64 x 64, block 8 | PASS |
| nw | `nw_kernel1` + `nw_kernel2`, anti-diagonal blocks | 64 x 64, block 16 | PASS |
| cfd | `memset_kernel`, `initialize_variables`, `compute_step_factor`, `compute_flux`, `time_step`, RK3 | 256 elements, 2 iterations | PASS |
| lavamd | `kernel_gpu_opencl` (particles staged through `__local`) | 8 boxes x 100 particles, 64-lane groups | PASS |
| btree | `findK` + `findRangeK` (one lane per key slot) | order 63, 512 keys, 32 queries | PASS |
| particlefilter | `particle_kernel` (naive resampling, double) | 512 particles | PASS |
| leukocyte | `GICOV_kernel` + `dilate_kernel` (detection stage) | 24 x 24 pixels, 7 circles x 150 points | PASS |
| hybridsort | `histogram1024Kernel`, `bucketcount`, `bucketprefixoffset`, `bucketsort` (bucket stage, warp-tagged `__local` atomics) | 2048 floats, 1024 buckets | PASS |
| srad | 6 kernels, `srad_kernel` first | 32 x 32 | FAIL (compiler, known-issues) |
| backprop | `bpnn_layerforward_ocl` + `bpnn_adjust_weights_ocl` (16x16 groups) | 64 -> 16 units | FAIL (group > maxvl, known-issues) |
| myocyte | `kernel_gpu_opencl` (ECC + 3x CaM, scalar per group) | 91 equations | FAIL (out of vs registers, known-issues) |
| dwt2d | `cl_fdwt53Kernel` | 64 x 64 | no .s (out of vs registers, known-issues) |
| heartwall | `kernel_gpu_opencl` | -- | no host (out of vs registers, known-issues) |

Not attempted: hybridsort's `mergesort.cl` (float4 vector ops), leukocyte's `track_ellipse_kernel`
(`atan()`, no hwacha-cc expansion), particlefilter's `particle_single.cl` (`image2d_t`) and
`particle_double.cl` (`ceil(double)`), and the `image2d_t` variants of leukocyte's kernels.

`known-issues/README.md` has the diagnosis of each failure. The compiler changes this directory
needed (switch lowering, 2-D NDRanges, ceil / mul24 / abs / usub.sat, get_global_size, a loop-header
skip jump, a SCEV zext hazard, constant-size memcpy expansion) are hwacha-compiler commit ea483c6;
the earlier changes of `../torch-*` (5b1dec2 .. 76ad080) are needed too.
