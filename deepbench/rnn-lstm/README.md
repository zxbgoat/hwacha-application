# rnn-lstm

DeepBench 的 **LSTM** 在 Hwacha 上运行：rnn_bench "lstm"：cuDNN CUDNN_LSTM（门序 i, f, o, g）、单层单向、SKIP_INPUT；每个 work-item 算自己单元的四个门点积，sigmoid / tanh 用 exp 展开。

DeepBench（baidu-research/DeepBench）本身没有内核，它在 `code/kernels/*.h` 的问题集上调用厂商库（cuDNN / cuBLAS / cuSPARSE / MKL ...）；`rnn-lstm.cl` 是按该库的语义为 hwacha-cc 写的 OpenCL 实现。hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `lstm_step_ct`。

问题规模：training 集的 3 个形状（隐层 512–2048）缩小（`rnn-lstm_main.c` 的 `shapes[]`，每项注明对应的 DeepBench 原形状与缩放比）。输入由固定种子的伪随机数生成；host 对每个形状先在 Rocket 标量核上跑 C 参考实现，再跑 Hwacha 内核，逐元素比对并打印两侧周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `rnn-lstm.s` | hwacha-cc 生成的汇编，入口 `net` |
| `rnn-lstm.cl` | 按 DeepBench 所调用库的语义写的 OpenCL 内核 |
| `rnn-lstm_main.c` | 裸机 host：问题形状表、输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、percentDiff、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make rnn-lstm          # 编译 -> rnn-lstm/rnn-lstm.riscv
make rnn-lstm.spike    # 在 Spike 上运行
make gen-rnn-lstm      # 从 .cl 重新生成汇编（需要 hwacha-cc）
```

## Spike 结果

| 形状（DeepBench 原形状与缩放） | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|
| hidden 32 batch 4 T 10 (training 512x16x25 (/16, /4, /2.5)) | 1,441,037 | 15,114 | 95x |
| hidden 64 batch 8 T 6 (training 2048x128x25 (/32, /16, /4)) | 5,025,443 | 32,846 | 153x |
| hidden 32 batch 4 T 12 (training 256x64x150 (/8, /16, /12)) | 1,725,881 | 18,128 | 95x |

结果：rnn-lstm **PASS**。
