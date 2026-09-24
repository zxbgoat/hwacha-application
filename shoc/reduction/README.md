# reduction

SHOC level1 `reduction` 在 Hwacha 上运行：**reduce（每个 work-group 把跨步切片累加进 __local，树形归约后写一个部分和，host 相加）+ reduceNoLocal（SHOC 给工作组大小为 1 的设备的备用内核）**。

内核文件（`reduction.cl`）是 SHOC 的原版，未做修改（多文件的基准由一个同名 .cl #include 汇总）；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `reduceNoLocal_ct`、`reduce_ct`，host 按 SHOC host 的启动顺序与工作组形状调用（__local 缓冲区是 host 数组：每个 work-group 独占一个 stripmine）。

问题规模：16384 个 float（i % 3），64 个 work-group（`reduction_main.c`）。输入按 SHOC host 的方式生成（固定种子）；host 先在 Rocket 标量核上跑参考实现，再跑 Hwacha 内核，按 SHOC 的判据比对并打印 PASS/FAIL 与周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `reduction.s` | hwacha-cc 生成的汇编，入口 `net` |
| `reduction.cl` | SHOC 原版 OpenCL 内核 |
| `reduction_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make reduction          # 编译 -> reduction/reduction.riscv
make reduction.spike    # 在 Spike 上运行
make gen-reduction      # 从 .cl 重新生成汇编（需要 hwacha-cc）
```

## Spike 结果

| 项 | 周期 |
|---|---|
| reduction scalar | 81,924 |
| reduction hwacha-cc | 1,330 |
| reduction hwacha-cc reduceNoLocal | 81,980 |

结果：reduction **PASS**。
