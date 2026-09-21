# btree

Rodinia `b+tree` 的 OpenCL 内核在 Hwacha 上运行：**findK（点查询）+ findRangeK（范围查询），B+ 树展平成 knode 数组，每个查询一个 work-group、每个 key 槽一个 lane**。

内核文件 `btree.cl` 是 Rodinia 3.1 的原版，唯一改动：kernel_gpu_opencl.cl 与 kernel_gpu_opencl_2.cl 合并为一个文件，第二个文件重复的结构体定义去掉、DEFAULT_ORDER_2 统一为 DEFAULT_ORDER；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `findK_ct`, `findRangeK_ct`，host 以 OpenCL host 传给 kernel 的同样参数调用它们。

问题规模：order 63（Rodinia 256 超过 maxvl），512 个 key，32 个查询（`btree_main.c` 中 `ORDER=63 NKEYS=512 NQ=32`）。输入由固定种子的伪随机数生成；host 先在 Rocket 标量核上跑一个参考实现，再跑 Hwacha 内核，逐元素比对并打印两者的周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `btree.s` | hwacha-cc 生成的汇编，入口 `net` |
| `btree.cl` | Rodinia 原版 OpenCL 内核（两个 .cl 合并为一个文件） |
| `btree_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make btree          # 编译 -> btree/btree.riscv
make btree.spike    # 在 Spike 上运行
make gen-btree      # 从 .cl 重新生成汇编（clang -> hwacha-cc）
```

## Spike 结果

| 内核 | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|
| btree | 78,554 | 1,509 | 52x |

结果：btree **PASS**。
