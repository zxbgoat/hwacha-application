# kmeans

Rodinia `kmeans` 的 OpenCL 内核在 Hwacha 上运行：**kmeans_swap（特征矩阵转置）+ kmeans_kernel_c（一步成员分配：每个点找最近的聚类中心）**。

内核文件 `kmeans.cl` 是 Rodinia 3.1 的原版，未做修改；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `kmeans_kernel_c_ct`, `kmeans_swap_ct`，host 以 OpenCL host 传给 kernel 的同样参数调用它们。

问题规模：1024 个点 x 8 个特征，5 个聚类（`kmeans_main.c` 中 `NP=1024 NF=8 NC=5`）。输入由固定种子的伪随机数生成；host 先在 Rocket 标量核上跑一个参考实现，再跑 Hwacha 内核，逐元素比对并打印两者的周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `kmeans.s` | hwacha-cc 生成的汇编，入口 `net` |
| `kmeans.cl` | Rodinia 原版 OpenCL 内核 |
| `kmeans_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make kmeans          # 编译 -> kmeans/kmeans.riscv
make kmeans.spike    # 在 Spike 上运行
make gen-kmeans      # 从 .cl 重新生成汇编（clang -> hwacha-cc）
```

## Spike 结果

| 内核 | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|
| kmeans_swap | 390,024 | 241 | 1618x |
| kmeans_c | 390,024 | 2,059 | 189x |

结果：kmeans_swap **PASS**，kmeans_c **PASS**。
