# rnn-gru

DeepBench 的 **GRU** 在 Hwacha 上运行：rnn_bench "gru"：cuDNN CUDNN_GRU（门序 r, z, h，h' = tanh(x + r * (R_h h + b_Rh) + b_Wh)）、单层单向、SKIP_INPUT。

DeepBench（baidu-research/DeepBench）本身没有内核，它在 `code/kernels/*.h` 的问题集上调用厂商库（cuDNN / cuBLAS / cuSPARSE / MKL ...）；`rnn-gru.cl` 是按该库的语义为 hwacha-cc 写的 OpenCL 实现。hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `gru_step_ct`。

问题规模：training 集的 3 个形状（隐层 1024–2816）缩小（`rnn-gru_main.c` 的 `shapes[]`，每项注明对应的 DeepBench 原形状与缩放比）。输入由固定种子的伪随机数生成；host 对每个形状先在 Rocket 标量核上跑 C 参考实现，再跑 Hwacha 内核，逐元素比对并打印两侧周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `rnn-gru.s` | hwacha-cc 生成的汇编，入口 `net` |
| `rnn-gru.cl` | 按 DeepBench 所调用库的语义写的 OpenCL 内核 |
| `rnn-gru_main.c` | 裸机 host：问题形状表、输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、percentDiff、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make rnn-gru          # 编译 -> rnn-gru/rnn-gru.riscv
make rnn-gru.spike    # 在 Spike 上运行
make gen-rnn-gru      # 从 .cl 重新生成汇编（需要 hwacha-cc）
```

## Spike 结果

| 形状（DeepBench 原形状与缩放） | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|
| hidden 44 batch 4 T 12 (training 2816x32x1500 (/64, /8, /125)) | 2,130,351 | 35,190 | 61x |
| hidden 64 batch 4 T 6 (training 2048x32x187 (/32, /8, /31)) | 1,978,213 | 32,850 | 60x |
| hidden 32 batch 8 T 10 (training 1024x64x1500 (/32, /8, /150)) | 2,151,344 | 15,130 | 142x |

结果：rnn-gru **PASS**。
