# conv

DeepBench 的 **卷积** 在 Hwacha 上运行：conv_bench：cuDNN 的前向（conv_fwd）、反向数据（conv_bwd_data）、反向权重（conv_bwd_filter），NCHW、互相关；直接卷积形式，每个输出元素一个 work-item，归约循环在 lane 内。

DeepBench（baidu-research/DeepBench）本身没有内核，它在 `code/kernels/*.h` 的问题集上调用厂商库（cuDNN / cuBLAS / cuSPARSE / MKL ...）；`conv.cl` 是按该库的语义为 hwacha-cc 写的 OpenCL 实现。hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `conv_bwd_data_ct`、`conv_bwd_filter_ct`、`conv_fwd_ct`。

问题规模：training 集中 VGG / ResNet / Inception / DeepSpeech 的 5 个层，通道与批缩小（`conv_main.c` 的 `shapes[]`，每项注明对应的 DeepBench 原形状与缩放比）。输入由固定种子的伪随机数生成；host 对每个形状先在 Rocket 标量核上跑 C 参考实现，再跑 Hwacha 内核，逐元素比对并打印两侧周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `conv.s` | hwacha-cc 生成的汇编，入口 `net` |
| `conv.cl` | 按 DeepBench 所调用库的语义写的 OpenCL 内核 |
| `conv_main.c` | 裸机 host：问题形状表、输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、percentDiff、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make conv          # 编译 -> conv/conv.riscv
make conv.spike    # 在 Spike 上运行
make gen-conv      # 从 .cl 重新生成汇编（需要 hwacha-cc）
```

## Spike 结果

| 形状（DeepBench 原形状与缩放） | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|
| 14x14x16 n2 k16 3x3 pad 1,1 stride 1,1 (VGG 14x14x512 n16 k512 3x3 (c,k /32, n /8)) | 72,473,449 | 5,824 | 12444x |
| 16x16x8 n2 k16 1x1 pad 0,0 stride 2,2 (ResNet 56x56x64 n8 k256 1x1 stride 2 (/3.5, c /8, k /16)) | 3,373,913 | 2,668 | 1265x |
| 14x14x12 n2 k8 5x5 pad 2,2 stride 1,1 (inception 28x28x192 n16 k32 5x5 pad 2 (/2, c /16, k /4)) | 66,947,921 | 3,938 | 17000x |
| 16x16x1 n2 k8 5x5 pad 0,0 stride 2,2 (DeepSpeech 700x161x1 n4 k32 20x5 stride 2 (5x5, /40, k /4)) | 1,724,803 | 1,314 | 1313x |
| 12x8x4 n3 k12 3x3 pad 1,1 stride 1,1 (DeepSpeech 120x12x32 n16 k64 3x3 (/10, c /8, k /5)) | 9,726,174 | 2,372 | 4100x |

结果：conv **PASS**。
