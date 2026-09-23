# rnn-vanilla

DeepBench 的 **vanilla RNN** 在 Hwacha 上运行：rnn_bench "vanilla"：cuDNN CUDNN_RNN_RELU、单层单向、CUDNN_SKIP_INPUT（输入直接进单元，无输入权重矩阵）；h_t = ReLU(x_t + R h_{t-1} + b)，每个时间步一次启动，每个 (batch, unit) 一个 work-item。

DeepBench（baidu-research/DeepBench）本身没有内核，它在 `code/kernels/*.h` 的问题集上调用厂商库（cuDNN / cuBLAS / cuSPARSE / MKL ...）；`rnn-vanilla.cl` 是按该库的语义为 hwacha-cc 写的 OpenCL 实现。hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `rnn_relu_step_ct`。

问题规模：training 集的 3 个形状（隐层 1760–2560）缩小（`rnn-vanilla_main.c` 的 `shapes[]`，每项注明对应的 DeepBench 原形状与缩放比）。输入由固定种子的伪随机数生成；host 对每个形状先在 Rocket 标量核上跑 C 参考实现，再跑 Hwacha 内核，逐元素比对并打印两侧周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `rnn-vanilla.s` | hwacha-cc 生成的汇编，入口 `net` |
| `rnn-vanilla.cl` | 按 DeepBench 所调用库的语义写的 OpenCL 内核 |
| `rnn-vanilla_main.c` | 裸机 host：问题形状表、输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、percentDiff、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make rnn-vanilla          # 编译 -> rnn-vanilla/rnn-vanilla.riscv
make rnn-vanilla.spike    # 在 Spike 上运行
make gen-rnn-vanilla      # 从 .cl 重新生成汇编（需要 hwacha-cc）
```

## Spike 结果

| 形状（DeepBench 原形状与缩放） | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|
| hidden 55 batch 4 T 10 (training 1760x16x50 (/32, /4, /5)) | 758,908 | 41,119 | 18x |
| hidden 64 batch 8 T 8 (training 2048x128x50 (/32, /16, /6)) | 1,633,618 | 43,261 | 38x |
| hidden 40 batch 2 T 12 (training 2560x32x50 (/64, /16, /4)) | 245,021 | 26,537 | 9x |

结果：rnn-vanilla **PASS**。
