# gemm

SHOC level1 `gemm` 在 Hwacha 上运行：**sgemmNN / sgemmNT（源自 MAGMA：16 x 4 的 work-group 算 64 x 16 的 C 分块，A 每 work-item 暂存 4 个元素，B 经 __local 16 x 17 分块），alpha = 1、beta = -1**。

内核文件（`gemm.cl`）是 SHOC 的原版，未做修改（多文件的基准由一个同名 .cl #include 汇总）；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `sgemmNN_ct`、`sgemmNT_ct`，host 按 SHOC host 的启动顺序与工作组形状调用（__local 缓冲区是 host 数组：每个 work-group 独占一个 stripmine）。

问题规模：128 x 128（SHOC 256），输入均匀分布于 [0.5, 2)（`gemm_main.c`）。输入按 SHOC host 的方式生成（固定种子）；host 先在 Rocket 标量核上跑参考实现，再跑 Hwacha 内核，按 SHOC 的判据比对并打印 PASS/FAIL 与周期数。

说明：已知限制：内核要求 64 lane 的 work-group（分块映射写死），在 Hwacha 上需要 <= 32 个向量寄存器，而 k 循环里 16 个 C 累加器 + 4 个 A 值 + 地址长期活跃（hwacha-cc 分配 65 个，maxvl 24），溢出器无法驱逐跨循环活跃的值；组被拆成三个 stripmine，经 barrier 的 B 分块读到错误的 lane，结果错误（host 报告 hwacha_vl_short）。

## 文件

| 文件 | 内容 |
|---|---|
| `gemm.s` | hwacha-cc 生成的汇编，入口 `net` |
| `gemm.cl` | SHOC 原版 OpenCL 内核 |
| `gemm_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make gemm          # 编译 -> gemm/gemm.riscv
make gemm.spike    # 在 Spike 上运行
make gen-gemm      # 从 .cl 重新生成汇编（需要 hwacha-cc）
```

## Spike 结果

| 项 | 周期 |
|---|---|
| gemm sgemmNN 128x128x128: scalar | 23,282,957 |
| gemm sgemmNN 128x128x128 / hwacha-cc | 8,901 |
| gemm sgemmNT 128x128x128: scalar | 23,282,957 |
| gemm sgemmNT 128x128x128 / hwacha-cc | 10,791 |

结果：gemm **FAIL**。
