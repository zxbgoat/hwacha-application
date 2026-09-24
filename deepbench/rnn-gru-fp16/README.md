# rnn-gru-fp16

DeepBench 的 **GRU（fp16）** 在 Hwacha 上运行：rnn_bench "gru" 的 "half" 精度：x / R / bW / bR / h 都是 half，门运算在 float 中进行，h_t 以 half 存回。

DeepBench（baidu-research/DeepBench）本身没有内核，它在 `code/kernels/*.h` 的问题集上调用厂商库（cuDNN / cuBLAS / cuSPARSE / MKL ...）；`rnn-gru-fp16.cl` 是按该库的语义为 hwacha-cc 写的 OpenCL 实现。hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `gru_step_ct`。

问题规模：与 rnn-gru 相同的 3 个形状（`rnn-gru-fp16_main.c` 的 `shapes[]`，每项注明对应的 DeepBench 原形状与缩放比）。输入由固定种子的伪随机数生成；host 对每个形状先在 Rocket 标量核上跑 C 参考实现，再跑 Hwacha 内核，逐元素比对并打印两侧周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `rnn-gru-fp16.s` | hwacha-cc 生成的汇编，入口 `net` |
| `rnn-gru-fp16.cl` | 按 DeepBench 所调用库的语义写的 OpenCL 内核 |
| `rnn-gru-fp16_main.c` | 裸机 host：问题形状表、输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、percentDiff、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make rnn-gru-fp16          # 编译 -> rnn-gru-fp16/rnn-gru-fp16.riscv
make rnn-gru-fp16.spike    # 在 Spike 上运行
make gen-rnn-gru-fp16      # 从 .cl 重新生成汇编（需要 hwacha-cc）
```

## Spike 结果

| 形状（DeepBench 原形状与缩放） | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|
| hidden 44 batch 4 T 12 (training 2816x32x1500 (/64, /8, /125)) | 2,225,203 | 35,158 | 63x |
| hidden 64 batch 4 T 6 (training 2048x32x187 (/32, /8, /31)) | 2,047,024 | 32,836 | 62x |
| hidden 32 batch 8 T 10 (training 1024x64x1500 (/32, /8, /150)) | 2,266,568 | 15,104 | 150x |

结果：rnn-gru-fp16 **PASS**。
