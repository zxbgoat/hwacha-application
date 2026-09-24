# triad

SHOC level1 `triad` 在 Hwacha 上运行：**Triad（Triad.cpp 内嵌的内核）：C = A + s*B，每元素一个 work-item，s = 1.75**。

内核文件（`triad.cl`）是 SHOC 的原版，未做修改（多文件的基准由一个同名 .cl #include 汇总）；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `Triad_ct`，host 按 SHOC host 的启动顺序与工作组形状调用（__local 缓冲区是 host 数组：每个 work-group 独占一个 stripmine）。

问题规模：64 / 128 / 256 KB 三种块大小（SHOC 到 16 MB）（`triad_main.c`）。输入按 SHOC host 的方式生成（固定种子）；host 先在 Rocket 标量核上跑参考实现，再跑 Hwacha 内核，按 SHOC 的判据比对并打印 PASS/FAIL 与周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `triad.s` | hwacha-cc 生成的汇编，入口 `net` |
| `triad.cl` | SHOC 原版 OpenCL 内核 |
| `triad_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make triad          # 编译 -> triad/triad.riscv
make triad.spike    # 在 Spike 上运行
make gen-triad      # 从 .cl 重新生成汇编（需要 hwacha-cc）
```

## Spike 结果

| 项 | 周期 |
|---|---|
| triad 64 KB block / : scalar | 131,086 |
| triad 64 KB block / hwacha-cc | 2,974 |
| triad 128 KB block / : scalar | 262,158 |
| triad 128 KB block / hwacha-cc | 5,918 |
| triad 256 KB block / : scalar | 524,302 |
| triad 256 KB block / hwacha-cc | 11,806 |

结果：triad **PASS**。
