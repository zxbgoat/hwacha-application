# lud

Rodinia `lud` 的 OpenCL 内核在 Hwacha 上运行：**lud_diagonal + lud_perimeter + lud_internal（分块原地 LU 分解，无 pivoting；每步对角块、周边块、剩余子矩阵，如 lud.cpp）**。

内核文件 `lud.cl` 是 Rodinia 3.1 的原版，未做修改；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `lud_diagonal_ct`, `lud_internal_ct`, `lud_perimeter_ct`，host 以 OpenCL host 传给 kernel 的同样参数调用它们。

问题规模：64x64 对角占优矩阵，块 8（`lud_main.c` 中 `BS=8 DIM=64`）。输入由固定种子的伪随机数生成；host 先在 Rocket 标量核上跑一个参考实现，再跑 Hwacha 内核，逐元素比对并打印两者的周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `lud.s` | hwacha-cc 生成的汇编，入口 `net` |
| `lud.cl` | Rodinia 原版 OpenCL 内核 |
| `lud_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make lud          # 编译 -> lud/lud.riscv
make lud.spike    # 在 Spike 上运行
make gen-lud      # 从 .cl 重新生成汇编（clang -> hwacha-cc）
```

## Spike 结果

| 内核 | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|
| lud | 824,927 | 28,807 | 29x |

结果：lud **PASS**。
