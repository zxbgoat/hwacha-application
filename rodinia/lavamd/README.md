# lavamd

Rodinia `lavaMD` 的 OpenCL 内核在 Hwacha 上运行：**kernel_gpu_opencl（相邻 box 粒子间的 N 体作用力，粒子经 __local 暂存；每个 box 一个 work-group）**。

内核文件 `lavamd.cl` 是 Rodinia 3.1 的原版，唯一改动：`NUMBER_THREADS` 的 #define 加了 #ifndef 以便由 Makefile 的 -D 覆盖；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `kernel_gpu_opencl_ct`，host 以 OpenCL host 传给 kernel 的同样参数调用它们。

问题规模：2x2x2=8 个 box，每个 100 个粒子；work-group 64（Rodinia 用 128，超过 maxvl 88）（`lavamd_main.c` 中 `B1D=2 PPB=100 NT=64`）。输入由固定种子的伪随机数生成；host 先在 Rocket 标量核上跑一个参考实现，再跑 Hwacha 内核，逐元素比对并打印两者的周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `lavamd.s` | hwacha-cc 生成的汇编，入口 `net` |
| `lavamd.cl` | Rodinia 原版 OpenCL 内核（lavaMD 的 NUMBER_THREADS 改为可由 -D 覆盖） |
| `lavamd_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make lavamd          # 编译 -> lavamd/lavamd.riscv
make lavamd.spike    # 在 Spike 上运行
make gen-lavamd      # 从 .cl 重新生成汇编（clang -> hwacha-cc）
```

## Spike 结果

| 内核 | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|
| lavamd | 75,779,326 | 293 | 258633x |

结果：lavamd **PASS**。
