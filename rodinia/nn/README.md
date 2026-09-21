# nn

Rodinia `nearestNeighbor` 的 OpenCL 内核在 Hwacha 上运行：**NearestNeighbor**。

内核文件 `nn.cl` 是 Rodinia 3.1 的原版，未做修改；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `NearestNeighbor_ct`，host 以 OpenCL host 传给 kernel 的同样参数调用它们。

问题规模：2048 条记录，每条一个 (纬度, 经度)，计算到查询点的距离（`nn_main.c` 中 `N=2048`）。输入由固定种子的伪随机数生成；host 先在 Rocket 标量核上跑一个参考实现，再跑 Hwacha 内核，逐元素比对并打印两者的周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `nn.s` | hwacha-cc 生成的汇编，入口 `net` |
| `nn.cl` | Rodinia 原版 OpenCL 内核 |
| `nn_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make nn          # 编译 -> nn/nn.riscv
make nn.spike    # 在 Spike 上运行
make gen-nn      # 从 .cl 重新生成汇编（clang -> hwacha-cc）
```

## Spike 结果

| 内核 | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|
| nn | 22,536 | 139 | 162x |

结果：nn **PASS**。
