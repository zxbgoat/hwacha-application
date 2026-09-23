# cholesky

PolyBench `cholesky` 在 Hwacha 上运行：**cholesky_kernel1/2/3：右视 Cholesky（列缩放、对角开方、尾部更新），k 在 host 迭代；与 4.2.1 的行视形式做同样的减法、同样的顺序，结果精确**。

PolyBench/GPU 没有这个 case 的 OpenCL 版本：`cholesky.cl` 是按 PolyBenchC-4.2.1 的 `kernel_cholesky` 循环嵌套为 hwacha-cc 改写的 OpenCL 内核，保持原公式与浮点运算顺序；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `cholesky_kernel1_ct`、`cholesky_kernel2_ct`、`cholesky_kernel3_ct`，host 按原 host 的顺序（依赖型算法把外层循环留在 host，每次迭代启动一个小内核）调用。

问题规模：48x48（`cholesky_main.c` 中 `N=48`），输入与标量按 PolyBenchC-4.2.1 的 `init_array`。host 先在 Rocket 标量核上跑 C 参考实现，再跑 Hwacha 内核，逐元素比对并打印 PASS/FAIL 与周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `cholesky.s` | hwacha-cc 生成的汇编，入口 `net` |
| `cholesky.cl` | 按 PolyBenchC-4.2.1 改写的 OpenCL 内核 |
| `cholesky_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、percentDiff、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make cholesky          # 编译 -> cholesky/cholesky.riscv
make cholesky.spike    # 在 Spike 上运行
make gen-cholesky      # 从 .cl 重新生成汇编（需要 hwacha-cc）
```

## Spike 结果

| 内核 | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|
| cholesky | 156,106 | 27,630 | 6x |

结果：cholesky **PASS**。
