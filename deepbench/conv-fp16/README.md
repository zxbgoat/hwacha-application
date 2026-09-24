# conv-fp16

DeepBench 的 **卷积（fp16）** 在 Hwacha 上运行：conv_bench "half" 精度：CUDNN_DATA_HALF 张量、float 计算；与 conv 相同的三个方向，半精度装载、单精度累加、写回时一次舍入。

DeepBench（baidu-research/DeepBench）本身没有内核，它在 `code/kernels/*.h` 的问题集上调用厂商库（cuDNN / cuBLAS / cuSPARSE / MKL ...）；`conv-fp16.cl` 是按该库的语义为 hwacha-cc 写的 OpenCL 实现。hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `conv_bwd_data_ct`、`conv_bwd_filter_ct`、`conv_fwd_ct`。

问题规模：与 conv 相同的 5 个层（`conv-fp16_main.c` 的 `shapes[]`，每项注明对应的 DeepBench 原形状与缩放比）。输入由固定种子的伪随机数生成；host 对每个形状先在 Rocket 标量核上跑 C 参考实现，再跑 Hwacha 内核，逐元素比对并打印两侧周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `conv-fp16.s` | hwacha-cc 生成的汇编，入口 `net` |
| `conv-fp16.cl` | 按 DeepBench 所调用库的语义写的 OpenCL 内核 |
| `conv-fp16_main.c` | 裸机 host：问题形状表、输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、percentDiff、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make conv-fp16          # 编译 -> conv-fp16/conv-fp16.riscv
make conv-fp16.spike    # 在 Spike 上运行
make gen-conv-fp16      # 从 .cl 重新生成汇编（需要 hwacha-cc）
```

## Spike 结果

| 形状（DeepBench 原形状与缩放） | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|
| 14x14x16 n2 k16 3x3 pad 1,1 stride 1,1 (VGG 14x14x512 n16 k512 3x3 (c,k /32, n /8)) | 81,674,908 | 5,457 | 14967x |
| 16x16x8 n2 k16 1x1 pad 0,0 stride 2,2 (ResNet 56x56x64 n8 k256 1x1 stride 2 (/3.5, c /8, k /16)) | 4,047,371 | 2,489 | 1626x |
| 14x14x12 n2 k8 5x5 pad 2,2 stride 1,1 (inception 28x28x192 n16 k32 5x5 pad 2 (/2, c /16, k /4)) | 75,133,036 | 3,687 | 20378x |
| 16x16x1 n2 k8 5x5 pad 0,0 stride 2,2 (DeepSpeech 700x161x1 n4 k32 20x5 stride 2 (5x5, /40, k /4)) | 1,966,797 | 1,233 | 1595x |
| 12x8x4 n3 k12 3x3 pad 1,1 stride 1,1 (DeepSpeech 120x12x32 n16 k64 3x3 (/10, c /8, k /5)) | 11,097,221 | 2,249 | 4934x |

结果：conv-fp16 **PASS**。
