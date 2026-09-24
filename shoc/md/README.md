# md

SHOC level1 `md` 在 Hwacha 上运行：**compute_lj_force：Lennard-Jones 力，每原子一个 work-item，邻居表转置存放（neighList[j*inum + idx]）；邻居表按 MD.cpp 的方式取最近的 maxNeighbors 个原子**。

内核文件（`md.cl`）是 SHOC 的原版，未做修改（多文件的基准由一个同名 .cl #include 汇总）；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `compute_lj_force_ct`，host 按 SHOC host 的启动顺序与工作组形状调用（__local 缓冲区是 host 数组：每个 work-group 独占一个 stripmine）。

问题规模：512 个原子（SHOC 12288），128 个邻居，cutsq 16、lj1 1.5、lj2 2.0（`md_main.c`）。输入按 SHOC host 的方式生成（固定种子）；host 先在 Rocket 标量核上跑参考实现，再跑 Hwacha 内核，按 SHOC 的判据比对并打印 PASS/FAIL 与周期数。

说明：float4 位置 / 力由 hwacha-cc 的 scalarizer 拆成标量。

## 文件

| 文件 | 内容 |
|---|---|
| `md.s` | hwacha-cc 生成的汇编，入口 `net` |
| `md.cl` | SHOC 原版 OpenCL 内核 |
| `md_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make md          # 编译 -> md/md.riscv
make md.spike    # 在 Spike 上运行
make gen-md      # 从 .cl 重新生成汇编（需要 hwacha-cc）
```

## Spike 结果

| 项 | 周期 |
|---|---|
| md: 6696 of 65536 pairs within the cutoff md hwacha-cc | 364 |
| md scalar | 1,649,531 |

结果：md **PASS**。
