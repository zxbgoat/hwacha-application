# adi

PolyBench `adi` 在 Hwacha 上运行：**adi_kernel1..6：交替方向隐式求解的行 / 列前代回代（N 在 .cl 中为编译期常量，Makefile 传 -DN=64）**。

内核文件 `adi.cl` 是 PolyBench/GPU 1.0 的原版 OpenCL 内核，未做修改；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `adi_kernel1_ct`、`adi_kernel2_ct`、`adi_kernel3_ct`、`adi_kernel4_ct`、`adi_kernel5_ct`、`adi_kernel6_ct`，host 按原 host 的顺序（依赖型算法把外层循环留在 host，每次迭代启动一个小内核）调用。

问题规模：64x64，2 步（`adi_main.c` 中 `N=64 TSTEPS=2`），输入与标量按 PolyBenchC-4.2.1 的 `init_array`。host 先在 Rocket 标量核上跑 C 参考实现，再跑 Hwacha 内核，逐元素比对并打印 PASS/FAIL 与周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `adi.s` | hwacha-cc 生成的汇编，入口 `net` |
| `adi.cl` | PolyBench/GPU 原版 OpenCL 内核 |
| `adi_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、percentDiff、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make adi          # 编译 -> adi/adi.riscv
make adi.spike    # 在 Spike 上运行
make gen-adi      # 从 .cl 重新生成汇编（需要 hwacha-cc）
```

## Spike 结果

| 内核 | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|
| adi | 502,430 | 21,092 | 24x |

结果：adi **PASS**。
