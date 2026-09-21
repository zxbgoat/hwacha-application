# Rodinia on Hwacha

The Rodinia OpenCL kernels that hwacha-cc compiles directly (`hwacha-cc/test/apps`), one directory per
application: the unmodified Rodinia kernel (`<app>.cl`, from `rodinia_3.1/opencl/<app>/`), hwacha-cc's
assembly of it (`<app>.s`, via `clang -x cl -emit-llvm` -> `hwacha-cc`), and a bare-metal host
(`<app>_main.c` + `common.h`) that builds a small random input, runs the scalar reference and the
Hwacha kernels, compares them and prints `<app> PASS` with both cycle counts.

Unlike the torch-* directories there is no PyTorch in the loop: the kernels are the original OpenCL
work-item programs, hwacha-cc maps work-items to Hwacha lanes and the host calls the generated
`<kernel>_ct` control-thread entry points with the same arguments the OpenCL host would pass.

| case | Rodinia kernel(s) | problem size | Hwacha vs scalar cycles |
|---|---|---|---|
| nn | nearestNeighbor (`NearestNeighbor`) | 2048 records | 139 vs 22,536 |
| kmeans | `kmeans_swap` (feature transpose) + `kmeans_kernel_c` (membership step) | 1024 points x 8 features, 5 clusters | 310 + 2,061 vs 390,024 |
| bfs | `BFS_1` + `BFS_2`, level-synchronous, iterated from the host | 2048 nodes, degree 4 | 4,696 vs 113,104 (10 levels) |
| pgain | streamcluster `memset_kernel` + `pgain_kernel` (cost of opening a center) | 1024 points x 8 dims, 4 centers | 540 vs 76,058 |
| pathfinder | `dynproc_kernel` (dynamic programming over rows) | 8 rows x 1024 cols, pyramid 2 | 1,038 vs 158,160 |

`make` builds all, `make run` runs all on Spike (5 PASS), `make <app>` / `make <app>.spike` for one,
`make gen-<app>` recompiles the assembly from the `.cl` (needs `~/miniforge3/bin/clang` and the
hwacha-cc build tree).

Not migrated from `hwacha-cc/test/apps`: `bfsp_main.c` (a bfs host without the reference check),
`bfsvar/` and `pfvar/` (kernel variants for RTL debugging), and the three Rodinia kernels that were
copied into `apps/rodinia/` but never given a host (`hotspot`, `lud`, `srad`: the .cl files need the
host's `BLOCK_SIZE` / `main.h` and do not compile on their own).
