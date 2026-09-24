# sort

SHOC level1 `sort` 在 Hwacha 上运行：**reduce + top_scan + bottom_scan：LSD 基数排序，4 位数字、8 趟，每趟三次启动，两个缓冲区乒乓**。

内核文件（`sort.cl`）是 SHOC 的原版，未做修改（多文件的基准由一个同名 .cl #include 汇总）；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `bottom_scan_ct`、`reduce_ct`、`top_scan_ct`，host 按 SHOC host 的启动顺序与工作组形状调用（__local 缓冲区是 host 数组：每个 work-group 独占一个 stripmine）。

问题规模：16384 个 uint（i % 16），64 个 work-group x 32（`sort_main.c`）。输入按 SHOC host 的方式生成（固定种子）；host 先在 Rocket 标量核上跑参考实现，再跑 Hwacha 内核，按 SHOC 的判据比对并打印 PASS/FAIL 与周期数。

说明：clang 以 -Xclang -disable-llvm-passes 编译：-O2 的 GlobalOpt 会把 top_scan 的 __local int s_seed 变成每 lane 的私有值。

## 文件

| 文件 | 内容 |
|---|---|
| `sort.s` | hwacha-cc 生成的汇编，入口 `net` |
| `sort.cl` | SHOC 原版 OpenCL 内核 |
| `sort_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make sort          # 编译 -> sort/sort.riscv
make sort.spike    # 在 Spike 上运行
make gen-sort      # 从 .cl 重新生成汇编（需要 hwacha-cc）
```

## Spike 结果

| 项 | 周期 |
|---|---|
| sort hwacha-cc | 42,342 |

结果：sort **PASS**。
