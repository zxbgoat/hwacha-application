# floyd-warshall

PolyBench `floyd-warshall` 在 Hwacha 上运行：**floyd_warshall_kernel：全源最短路，每个中间点 k 一次二维启动（int）**。

PolyBench/GPU 没有这个 case 的 OpenCL 版本：`floyd-warshall.cl` 是按 PolyBenchC-4.2.1 的 `kernel_floyd_warshall` 循环嵌套为 hwacha-cc 改写的 OpenCL 内核，保持原公式与浮点运算顺序；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `floyd_warshall_kernel_ct`，host 按原 host 的顺序（依赖型算法把外层循环留在 host，每次迭代启动一个小内核）调用。

问题规模：48x48（`floyd-warshall_main.c` 中 `N=48`），输入与标量按 PolyBenchC-4.2.1 的 `init_array`。host 先在 Rocket 标量核上跑 C 参考实现，再跑 Hwacha 内核，逐元素比对并打印 PASS/FAIL 与周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `floyd-warshall.s` | hwacha-cc 生成的汇编，入口 `net` |
| `floyd-warshall.cl` | 按 PolyBenchC-4.2.1 改写的 OpenCL 内核 |
| `floyd-warshall_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、percentDiff、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make floyd-warshall          # 编译 -> floyd-warshall/floyd-warshall.riscv
make floyd-warshall.spike    # 在 Spike 上运行
make gen-floyd-warshall      # 从 .cl 重新生成汇编（需要 hwacha-cc）
```

## Spike 结果

| 内核 | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|
| floyd-warshall | 1,435,159 | 33,331 | 43x |

结果：floyd-warshall **PASS**。
