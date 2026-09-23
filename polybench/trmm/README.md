# trmm

PolyBench `trmm` 在 Hwacha 上运行：**trmm_kernel：B = alpha*A^T*B，A 单位下三角；读未修改的 B 写到另一缓冲区**。

PolyBench/GPU 没有这个 case 的 OpenCL 版本：`trmm.cl` 是按 PolyBenchC-4.2.1 的 `kernel_trmm` 循环嵌套为 hwacha-cc 改写的 OpenCL 内核，保持原公式与浮点运算顺序；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `trmm_kernel_ct`，host 按原 host 的顺序（依赖型算法把外层循环留在 host，每次迭代启动一个小内核）调用。

问题规模：32x40（`trmm_main.c` 中 `M=32 N=40`），输入与标量按 PolyBenchC-4.2.1 的 `init_array`。host 先在 Rocket 标量核上跑 C 参考实现，再跑 Hwacha 内核，逐元素比对并打印 PASS/FAIL 与周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `trmm.s` | hwacha-cc 生成的汇编，入口 `net` |
| `trmm.cl` | 按 PolyBenchC-4.2.1 改写的 OpenCL 内核 |
| `trmm_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、percentDiff、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make trmm          # 编译 -> trmm/trmm.riscv
make trmm.spike    # 在 Spike 上运行
make gen-trmm      # 从 .cl 重新生成汇编（需要 hwacha-cc）
```

## Spike 结果

| 内核 | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|
| trmm | 175,317 | 426 | 412x |

结果：trmm **PASS**。
