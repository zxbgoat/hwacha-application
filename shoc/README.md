# shoc

SHOC（Scalable HeterOgeneous Computing benchmark suite，vetter/shoc）的 OpenCL 基准在 Hwacha 上运行：
每个基准一个目录，含 SHOC 原版内核文件（`.cl`，未做修改；多文件的基准由一个同名 `.cl` 用 `#include`
汇总）、hwacha-cc 生成的汇编（`.s`）、裸机 host（`<bench>_main.c`：按 SHOC host 的方式生成输入、按同样的
启动顺序与工作组形状调用内核、与 Rocket 标量核上的参考比对、打印 PASS/FAIL 与两侧周期数）和 `common.h`。
内核里的 `__local` 缓冲区是 host 数组：hwacha-cc 让每个 work-group 独占一个 stripmine，所有组共用一份。

| case | SHOC | 内核 | 规模（缩小后） | Hwacha |
|---|---|---|---|---|
| triad | level1 Triad | `Triad`（Triad.cpp 内嵌） | 64–256 KB 块 | PASS |
| reduction | level1 Reduction | `reduce`, `reduceNoLocal` | 16384，64 组 | PASS |
| scan | level1 Scan | `reduce`, `top_scan`, `bottom_scan` | 16384，64 组 x 64 | PASS |
| sort | level1 Sort | `reduce`, `top_scan`, `bottom_scan`（基数排序，8 趟） | 16384，64 组 x 32 | PASS |
| spmv | level1 Spmv | `spmv_csr_scalar_kernel`, `spmv_csr_vector_kernel`, `spmv_ellpackr_kernel` | 256 x 256，1% 非零 | PASS |
| md | level1 MD | `compute_lj_force` | 512 原子 x 128 邻居 | PASS |
| md5hash | level1 MD5Hash | `FindKeyWithDigest_Kernel` | 4 字节 x 10 值 | PASS |
| stencil2d | level1 Stencil2D | `StencilKernel`, `CopyRect` | 64 x 64 内部，10 次迭代 | PASS |
| bfs | level1 BFS | `BFS_kernel_warp`（bfs_iiit.cl；bfs_uiuc_spill.cl 编入但无 host） | 2048 顶点，度 2 | PASS |
| fft | level1 FFT | `fft1D_512`, `ifft1D_512`, `chk1D_512` | 8 个 512 点 FFT | PASS |
| gemm | level1 GEMM | `sgemmNN`, `sgemmNT` | 128 x 128 | FAIL（见下） |
| s3d | level2 S3D | 27 个内核（gr_base、ratt*、rdsmh、ratx*、qssa*、rdwdot*） | 64 个网格点 | PASS |

**未迁移**：level0（BusSpeedDownload / Readback、DeviceMemory、KernelCompile、MaxFlops、QueueDelay）——
它们测 PCIe 带宽、编译时间、队列延迟和访存 / 峰值 FLOPS，内核在 host 里按参数拼字符串生成，没有可校验的
计算结果；两个 SHOC 提供双精度变体的基准（`K_DOUBLE_PRECISION`）只跑了单精度；`tpmpi` / `epmpi` 多进程版本。

**gemm 的限制**：SHOC 的 sgemm 内核（源自 MAGMA）把 16 x 4 的 work-group 映射到 64 x 16 的 C 分块，B 分块经
`__local` 内存在 barrier 之间共享，所以 work-group 必须是 64 个 lane；Hwacha 的向量长度是
`maxvl = 8 * 256 / 向量寄存器数`，64 个 lane 要求内核不超过 32 个向量寄存器，而 k 循环里 16 个 C 累加器、4 个
A 值和分块地址长期活跃，hwacha-cc 分配了 65 个（maxvl 24），`-vregs 32` 下溢出器找不到可驱逐的值（跨循环
活跃的值不能在循环中途溢出）。组被拆成三个 stripmine 后 barrier 两侧的 lane 对不上，结果错误；host 报告
mismatch 与 `hwacha_vl_short`。汇编与 host 保留，README 记录。

**hwacha-cc 为 SHOC 补的功能**（hwacha-compiler 提交见 git log）：`get_num_groups(0)`；OpenCL 向量类型
（`float2` / `float4` / `uint4`：fft、md、scan、sort）经 LLVM scalarizer 拆成标量；`sinf` / `cosf`（fft 的旋转
因子）、`exp10f`（s3d）的展开；`llvm.fshl` / `fshr`（md5 的循环左移）的展开；两个 codegen 修复（flattenNDRange
在擦除的调用处插入指令、由流寄存器承载指针的 phi 不再申请寄存器）。sort 要用 `-Xclang -disable-llvm-passes`
编译：clang -O2 的 GlobalOpt 把 `top_scan` 里的 `__local int s_seed` 私有化成每 lane 的值。

```
make            # 编译全部 -> <bench>/<bench>.riscv
make run        # 全部在 Spike 上运行，每个 case 一行 PASS/FAIL
make <bench>    # 编译一个
make <bench>.spike
make gen-<bench> # 从 .cl 重新生成汇编（需要 hwacha-cc）
```
