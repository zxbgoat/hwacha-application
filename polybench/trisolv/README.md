# trisolv

PolyBench `trisolv` 在 Hwacha 上运行：**trisolv_kernel：按列的前代（第 i 行的 work-item 完成 x[i]，其余行减去第 i 列），每行一次启动；减法顺序与 4.2.1 相同，结果精确**。

PolyBench/GPU 没有这个 case 的 OpenCL 版本：`trisolv.cl` 是按 PolyBenchC-4.2.1 的 `kernel_trisolv` 循环嵌套为 hwacha-cc 改写的 OpenCL 内核，保持原公式与浮点运算顺序；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `trisolv_kernel_ct`，host 按原 host 的顺序（依赖型算法把外层循环留在 host，每次迭代启动一个小内核）调用。

问题规模：64x64（`trisolv_main.c` 中 `N=64`），输入与标量按 PolyBenchC-4.2.1 的 `init_array`。host 先在 Rocket 标量核上跑 C 参考实现，再跑 Hwacha 内核，逐元素比对并打印 PASS/FAIL 与周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `trisolv.s` | hwacha-cc 生成的汇编，入口 `net` |
| `trisolv.cl` | 按 PolyBenchC-4.2.1 改写的 OpenCL 内核 |
| `trisolv_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、percentDiff、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make trisolv          # 编译 -> trisolv/trisolv.riscv
make trisolv.spike    # 在 Spike 上运行
make gen-trisolv      # 从 .cl 重新生成汇编（需要 hwacha-cc）
```

## Spike 结果

| 内核 | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|
| trisolv | 14,947 | 3,722 | 4x |

结果：trisolv **PASS**。
