# scan

SHOC level1 `scan` 在 Hwacha 上运行：**reduce + top_scan + bottom_scan：三段前缀和（每组区域求和、单组扫描组和、每组以组和为种子按 4 元素向量扫描），按 Scan.cpp 的顺序启动**。

内核文件（`scan.cl`）是 SHOC 的原版，未做修改（多文件的基准由一个同名 .cl #include 汇总）；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `bottom_scan_ct`、`reduce_ct`、`top_scan_ct`，host 按 SHOC host 的启动顺序与工作组形状调用（__local 缓冲区是 host 数组：每个 work-group 独占一个 stripmine）。

问题规模：16384 个 float，64 个 work-group x 64（`scan_main.c`）。输入按 SHOC host 的方式生成（固定种子）；host 先在 Rocket 标量核上跑参考实现，再跑 Hwacha 内核，按 SHOC 的判据比对并打印 PASS/FAIL 与周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `scan.s` | hwacha-cc 生成的汇编，入口 `net` |
| `scan.cl` | SHOC 原版 OpenCL 内核 |
| `scan_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make scan          # 编译 -> scan/scan.riscv
make scan.spike    # 在 Spike 上运行
make gen-scan      # 从 .cl 重新生成汇编（需要 hwacha-cc）
```

## Spike 结果

| 项 | 周期 |
|---|---|
| scan scalar | 98,311 |
| scan hwacha-cc | 2,736 |

结果：scan **PASS**。
