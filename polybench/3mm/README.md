# 3mm

PolyBench `3mm` 在 Hwacha 上运行：**mm3_kernel1/2/3：E = A*B，F = C*D，G = E*F**。

内核文件 `3mm.cl` 是 PolyBench/GPU 1.0 的原版 OpenCL 内核，未做修改；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `mm3_kernel1_ct`、`mm3_kernel2_ct`、`mm3_kernel3_ct`，host 按原 host 的顺序（依赖型算法把外层循环留在 host，每次迭代启动一个小内核）调用。

问题规模：32^5（`3mm_main.c` 中 `NI=NJ=NK=NL=NM=32`），输入与标量按 PolyBenchC-4.2.1 的 `init_array`。host 先在 Rocket 标量核上跑 C 参考实现，再跑 Hwacha 内核，逐元素比对并打印 PASS/FAIL 与周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `3mm.s` | hwacha-cc 生成的汇编，入口 `net` |
| `3mm.cl` | PolyBench/GPU 原版 OpenCL 内核 |
| `3mm_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、percentDiff、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make 3mm          # 编译 -> 3mm/3mm.riscv
make 3mm.spike    # 在 Spike 上运行
make gen-3mm      # 从 .cl 重新生成汇编（需要 hwacha-cc）
```

## Spike 结果

| 内核 | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|
| 3mm | 615,093 | 16,749 | 37x |

结果：3mm **PASS**。
