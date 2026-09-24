# spmv

SHOC level1 `spmv` 在 Hwacha 上运行：**spmv_csr_scalar_kernel（每行一个 work-item）、spmv_csr_vector_kernel（每行 vecWidth 个 work-item，__local 部分和树形归约）、spmv_ellpackr_kernel（列主序 ELLPACK-R）**。

内核文件（`spmv.cl`）是 SHOC 的原版，未做修改（多文件的基准由一个同名 .cl #include 汇总）；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `spmv_csr_scalar_kernel_ct`、`spmv_csr_vector_kernel_ct`、`spmv_ellpackr_kernel_ct`，host 按 SHOC host 的启动顺序与工作组形状调用（__local 缓冲区是 host 数组：每个 work-group 独占一个 stripmine）。

问题规模：256 x 256、1% 非零（util.h initRandomMatrix），值与 x 均匀分布于 [0, 10)（`spmv_main.c`）。输入按 SHOC host 的方式生成（固定种子）；host 先在 Rocket 标量核上跑参考实现，再跑 Hwacha 内核，按 SHOC 的判据比对并打印 PASS/FAIL 与周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `spmv.s` | hwacha-cc 生成的汇编，入口 `net` |
| `spmv.cl` | SHOC 原版 OpenCL 内核 |
| `spmv_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make spmv          # 编译 -> spmv/spmv.riscv
make spmv.spike    # 在 Spike 上运行
make gen-spmv      # 从 .cl 重新生成汇编（需要 hwacha-cc）
```

## Spike 结果

| 项 | 周期 |
|---|---|
| spmv scalar | 10,264 |
| spmv hwacha-cc csr_scalar | 91 |
| spmv hwacha-cc csr_vector | 696 |
| spmv hwacha-cc ellpackr | 87 |

结果：spmv **PASS**。
