# polybench

PolyBench 的全部 30 个 case（PolyBenchC-4.2.1 的清单）加上 PolyBench/GPU 1.0 里 4.2.1 已去掉的两个卷积
case，在 Hwacha 上运行：每个 case 一个目录，含 OpenCL 内核（`.cl`）、hwacha-cc 生成的汇编（`.s`）、裸机
host（`<case>_main.c`）和 `common.h`。host 构造输入，先在 Rocket 标量核上跑 C 参考实现，再跑 Hwacha 内核，
逐元素比对并打印 `<case> PASS/FAIL` 与两侧周期数。没有 PyTorch 参与：内核由 clang `-x cl` 编译成 LLVM IR
再交给 hwacha-cc。

内核有两个来源：

- **PolyBench/GPU 1.0 的原版 OpenCL 内核，未做修改**（20 个，来自 `sgrauerg/polybenchGpu`）。host 按原
  host 的启动顺序与工作组形状调用（原 host 用 32x8 = 256 lane 的组，这里用 8x8 或 64x1，以落在 Hwacha
  的 `maxvl` 内）。
- **按 PolyBenchC-4.2.1 改写的 OpenCL 内核**（12 个：4.2.1 新增而 PolyBench/GPU 没有的 case）。改写保持
  4.2.1 `kernel_*` 的公式和浮点运算顺序；带依赖的算法沿用 PolyBench/GPU 对 `lu` / `gramschmidt` 的做法，
  外层依赖循环留在 host、每次迭代启动一个小内核（`cholesky` / `ludcmp` 用右视形式，`trisolv` / `ludcmp` 的
  代入按列进行，`nussinov` 按对角线、`seidel-2d` 按反对角线波前）。

输入与标量按 4.2.1 的 `init_array`（PolyBench/GPU 的 host 与之不同处以 4.2.1 为准）；问题规模缩到 16–128，
每个 case 在 Spike 上几秒内跑完。目录名按 4.2.1；两个卷积 case 叫 `2dconv` / `3dconv`。

```
make            # 编译全部 -> <case>/<case>.riscv
make run        # 全部在 Spike 上运行，每个 case 一行 PASS/FAIL
make <case>     # 编译一个
make <case>.spike
make gen-<case> # 从 .cl 重新生成汇编（需要 hwacha-cc）
```

| case | 来源 | 内核 | 问题规模 | Hwacha |
|---|---|---|---|---|
| 2dconv | GPU | `Convolution2D_kernel` | 64x64 | PASS |
| 3dconv | GPU | `Convolution3D_kernel`，每个 i 平面一次启动 | 16x32x32 | PASS |
| 2mm | GPU | `mm2_kernel1` + `mm2_kernel2` | 32 | PASS |
| 3mm | GPU | `mm3_kernel1/2/3` | 32 | PASS |
| adi | GPU | `adi_kernel1..6`（`-DN=64`） | 64, 2 步 | PASS |
| atax | GPU | `atax_kernel1` + `atax_kernel2` | 64 | PASS |
| bicg | GPU | `bicgKernel1` + `bicgKernel2` | 64 | PASS |
| corr | GPU | `mean_kernel`, `std_kernel`, `reduce_kernel`, `corr_kernel` | 40x32 | PASS |
| covar | GPU | `mean_kernel`, `reduce_kernel`, `covar_kernel` | 40x32 | PASS |
| fdtd-2d | GPU | `fdtd_kernel1/2/3` | 48x48, 4 步 | PASS |
| gemm | GPU | `gemm` | 48 | PASS |
| gemver | GPU | `gemver_kernel1/2/3` | 64 | PASS |
| gesummv | GPU | `gesummv_kernel` | 64 | PASS |
| gramschmidt | GPU | `gramschmidt_kernel1/2/3`，k 在 host | 48x48 | PASS |
| jacobi-1d | GPU | `runJacobi1D_kernel1/2` | 128, 8 步 | PASS |
| jacobi-2d | GPU | `runJacobi2D_kernel1/2` | 64, 4 步 | PASS |
| lu | GPU | `lu_kernel1/2`，k 在 host | 64 | PASS |
| mvt | GPU | `mvt_kernel1/2` | 64 | PASS |
| syr2k | GPU | `syr2k_kernel` | 48 | PASS |
| syrk | GPU | `syrk_kernel` | 48 | PASS |
| symm | 4.2.1 | `symm_kernel` | 32x40 | PASS |
| trmm | 4.2.1 | `trmm_kernel` | 32x40 | PASS |
| doitgen | 4.2.1 | `doitgen_kernel` | 16^3 | PASS |
| cholesky | 4.2.1 | `cholesky_kernel1/2/3`，k 在 host | 48 | PASS |
| durbin | 4.2.1 | `durbin_kernel1/2/3`，k 在 host | 64 | PASS |
| ludcmp | 4.2.1 | `ludcmp_kernel1..4`，k / i 在 host | 48 | PASS |
| trisolv | 4.2.1 | `trisolv_kernel`，i 在 host | 64 | PASS |
| deriche | 4.2.1 | `deriche_kernel1..5` | 32x32 | PASS |
| floyd-warshall | 4.2.1 | `floyd_warshall_kernel`，k 在 host | 48 | PASS |
| nussinov | 4.2.1 | `nussinov_kernel`，按对角线 | 48 | PASS |
| heat-3d | 4.2.1 | `heat_3d_kernel`，每步两次 | 16^3, 4 步 | PASS |
| seidel-2d | 4.2.1 | `seidel_2d_kernel`，按反对角线 | 32, 2 步 | PASS |

全部 32 个 case 与参考实现的结果逐元素精确相等（`max|diff| = 0`，包括 `ludcmp` 回代减法顺序相反的那一步）。
比对沿用 rodinia 的 `check_f`（相对误差阈值），`common.h` 另提供 PolyBench 自己的 `percentDiff` /
`check_percent`。周期数见各 case 目录的 README。
