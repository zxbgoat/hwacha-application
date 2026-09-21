# gaussian

Rodinia `gaussian` 的 OpenCL 内核在 Hwacha 上运行：**Fan1（每个 pivot 的乘子，1-D）+ Fan2（行更新，2-D NDRange），逐 pivot 从 host 驱动，如 gaussianElim.cpp**。

内核文件 `gaussian.cl` 是 Rodinia 3.1 的原版，未做修改；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `Fan1_ct`, `Fan2_ct`，host 以 OpenCL host 传给 kernel 的同样参数调用它们。

问题规模：64x64 对角占优线性系统（`gaussian_main.c` 中 `N=64 LS=8`）。输入由固定种子的伪随机数生成；host 先在 Rocket 标量核上跑一个参考实现，再跑 Hwacha 内核，逐元素比对并打印两者的周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `gaussian.s` | hwacha-cc 生成的汇编，入口 `net` |
| `gaussian.cl` | Rodinia 原版 OpenCL 内核 |
| `gaussian_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make gaussian          # 编译 -> gaussian/gaussian.riscv
make gaussian.spike    # 在 Spike 上运行
make gen-gaussian      # 从 .cl 重新生成汇编（clang -> hwacha-cc）
```

## Spike 结果

| 内核 | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|
| gaussian | 1,004,509 | 36,373 | 28x |

结果：gaussian **PASS**。
