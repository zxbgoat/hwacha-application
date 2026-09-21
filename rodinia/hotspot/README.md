# hotspot

Rodinia `hotspot` 的 OpenCL 内核在 Hwacha 上运行：**hotspot（2-D 热传导 stencil，BLOCK_SIZE x BLOCK_SIZE 分块，每次启动做 pyramid 高度步；hotspot.c 的 compute_tran_temp）**。

内核文件 `hotspot.cl` 是 Rodinia 3.1 的原版，未做修改；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `hotspot_ct`，host 以 OpenCL host 传给 kernel 的同样参数调用它们。

问题规模：64x64 网格，4 步，每次启动 2 步；块 8x8（Rodinia 用 16，16x16=256 lane 超过 Hwacha 给这个内核的 maxvl）（`hotspot_main.c` 中 `BS=8 ROWS=64 COLS=64 PYR=2 TOTAL=4`）。输入由固定种子的伪随机数生成；host 先在 Rocket 标量核上跑一个参考实现，再跑 Hwacha 内核，逐元素比对并打印两者的周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `hotspot.s` | hwacha-cc 生成的汇编，入口 `net` |
| `hotspot.cl` | Rodinia 原版 OpenCL 内核 |
| `hotspot_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make hotspot          # 编译 -> hotspot/hotspot.riscv
make hotspot.spike    # 在 Spike 上运行
make gen-hotspot      # 从 .cl 重新生成汇编（clang -> hwacha-cc）
```

## Spike 结果

| 内核 | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|
| hotspot | 596,028 | 11,494 | 52x |

结果：hotspot **PASS**。
