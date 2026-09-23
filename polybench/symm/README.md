# symm

PolyBench `symm` 在 Hwacha 上运行：**symm_kernel：C = alpha*A*B + beta*C，A 对称（存下三角），每个 C 元素一个 work-item，按 kernel_symm 的运算顺序求和**。

PolyBench/GPU 没有这个 case 的 OpenCL 版本：`symm.cl` 是按 PolyBenchC-4.2.1 的 `kernel_symm` 循环嵌套为 hwacha-cc 改写的 OpenCL 内核，保持原公式与浮点运算顺序；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `symm_kernel_ct`，host 按原 host 的顺序（依赖型算法把外层循环留在 host，每次迭代启动一个小内核）调用。

问题规模：32x40（`symm_main.c` 中 `M=32 N=40`），输入与标量按 PolyBenchC-4.2.1 的 `init_array`。host 先在 Rocket 标量核上跑 C 参考实现，再跑 Hwacha 内核，逐元素比对并打印 PASS/FAIL 与周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `symm.s` | hwacha-cc 生成的汇编，入口 `net` |
| `symm.cl` | 按 PolyBenchC-4.2.1 改写的 OpenCL 内核 |
| `symm_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、percentDiff、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make symm          # 编译 -> symm/symm.riscv
make symm.spike    # 在 Spike 上运行
make gen-symm      # 从 .cl 重新生成汇编（需要 hwacha-cc）
```

## Spike 结果

| 内核 | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|
| symm | 223,072 | 434 | 514x |

结果：symm **PASS**。
