# pgain

Rodinia `streamcluster` 的 OpenCL 内核在 Hwacha 上运行：**memset_kernel + pgain_kernel（对每个点计算打开中心 x 的代价）**。

内核文件 `pgain.cl` 是 Rodinia 3.1 的原版，未做修改；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 ``，host 以 OpenCL host 传给 kernel 的同样参数调用它们。

问题规模：1024 个点 x 8 维，4 个中心（`pgain_main.c` 中 `NUM=1024 DIM=8 K=4 GROUP=256`）。输入由固定种子的伪随机数生成；host 先在 Rocket 标量核上跑一个参考实现，再跑 Hwacha 内核，逐元素比对并打印两者的周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `pgain.s` | hwacha-cc 生成的汇编，入口 `net` |
| `pgain.cl` | Rodinia 原版 OpenCL 内核 |
| `pgain_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT 宏与 newlib 的 __errno 桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make pgain          # 编译 -> pgain/pgain.riscv
make pgain.spike    # 在 Spike 上运行
make gen-pgain      # 从 .cl 重新生成汇编（clang -> hwacha-cc）
```

## Spike 结果

| 内核 | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|
| pgain | 76,058 | 540 | 141x |

结果：pgain **PASS**。
