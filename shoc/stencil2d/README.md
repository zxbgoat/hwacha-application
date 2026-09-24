# stencil2d

SHOC level1 `stencil2d` 在 Hwacha 上运行：**StencilKernel（每个 work-item 从 __local 的带 halo 分块算 LROWS 行）+ CopyRect（把左右 halo 列搬到新缓冲区），每次迭代交换缓冲区**。

内核文件（`stencil2d.cl`）是 SHOC 的原版，未做修改（多文件的基准由一个同名 .cl #include 汇总）；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `CopyRect_ct`、`StencilKernel_ct`，host 按 SHOC host 的启动顺序与工作组形状调用（__local 缓冲区是 host 数组：每个 work-group 独占一个 stripmine）。

问题规模：66 x 66（64 x 64 内部），10 次迭代（SHOC 1000），权重 0.25 / 0.15 / 0.05（`stencil2d_main.c`）。输入按 SHOC host 的方式生成（固定种子）；host 先在 Rocket 标量核上跑参考实现，再跑 Hwacha 内核，按 SHOC 的判据比对并打印 PASS/FAIL 与周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `stencil2d.s` | hwacha-cc 生成的汇编，入口 `net` |
| `stencil2d.cl` | SHOC 原版 OpenCL 内核 |
| `stencil2d_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make stencil2d          # 编译 -> stencil2d/stencil2d.riscv
make stencil2d.spike    # 在 Spike 上运行
make gen-stencil2d      # 从 .cl 重新生成汇编（需要 hwacha-cc）
```

## Spike 结果

| 项 | 周期 |
|---|---|
| stencil2d scalar | 1,112,443 |
| stencil2d hwacha-cc | 175,518 |

结果：stencil2d **PASS**。
