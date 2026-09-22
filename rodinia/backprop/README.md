# backprop

Rodinia `backprop` 的 OpenCL 内核在 Hwacha 上运行：**bpnn_layerforward_ocl（16x16 work-group，乘积 + 组内树形归约得部分和）+ bpnn_adjust_weights_ocl（权重更新），如 backprop_ocl.cpp**。

内核文件 `backprop.cl` 是 Rodinia 3.1 的原版，未做修改；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `bpnn_adjust_weights_ocl_ct`, `bpnn_layerforward_ocl_ct`，host 以 OpenCL host 传给 kernel 的同样参数调用它们。

问题规模：64 输入单元 -> 16 隐层单元（`backprop_main.c` 中 `IN=64 HID=16`）。输入由固定种子的伪随机数生成；host 先在 Rocket 标量核上跑一个参考实现，再跑 Hwacha 内核，逐元素比对并打印两者的周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `backprop.s` | hwacha-cc 生成的汇编，入口 `net` |
| `backprop.cl` | Rodinia 原版 OpenCL 内核 |
| `backprop_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make backprop          # 编译 -> backprop/backprop.riscv
make backprop.spike    # 在 Spike 上运行
make gen-backprop      # 从 .cl 重新生成汇编（clang -> hwacha-cc）
```

## Spike 结果

| 内核 | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|
| backprop | 54,716 | 291 | 188x |

结果：backprop **PASS**。
