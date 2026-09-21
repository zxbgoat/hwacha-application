# hybridsort

Rodinia `hybridsort（桶排序阶段）` 的 OpenCL 内核在 Hwacha 上运行：**histogram1024Kernel（warp-tag 的 __local 原子直方图）、bucketcount（每个元素的桶号与槽位）、bucketprefixoffset、bucketsort（散射）；pivot 与桶起点在 host 上算，如 bucketsort.c**。

内核文件 `hybridsort.cl` 是 Rodinia 3.1 的原版，唯一改动：histogram1024.cl 与 bucketsort_kernels.cl 合并为一个文件；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `bucketcount_ct`, `bucketprefixoffset_ct`, `bucketsort_ct`, `histogram1024Kernel_ct`，host 以 OpenCL host 传给 kernel 的同样参数调用它们。

问题规模：2048 个 float，1024 个桶；直方图 6144/96、count 与 sort 32-lane 组、prefix 1024/128（`hybridsort_main.c` 中 `N=2048 DIVISIONS=1024`）。输入由固定种子的伪随机数生成；host 先在 Rocket 标量核上跑一个参考实现，再跑 Hwacha 内核，逐元素比对并打印两者的周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `hybridsort.s` | hwacha-cc 生成的汇编，入口 `net` |
| `hybridsort.cl` | Rodinia 原版 OpenCL 内核（两个 .cl 合并为一个文件） |
| `hybridsort_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make hybridsort          # 编译 -> hybridsort/hybridsort.riscv
make hybridsort.spike    # 在 Spike 上运行
make gen-hybridsort      # 从 .cl 重新生成汇编（clang -> hwacha-cc）
```

## Spike 结果

| 内核 | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|
| hybridsort | 259,189 | 53,053 | 5x |

结果：hybridsort **PASS**。
