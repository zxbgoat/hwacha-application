# gemm-int8

DeepBench 的 **GEMM（int8 推理）** 在 Hwacha 上运行：gemm_bench inference int8：cublasGemmEx CUDA_R_8I x CUDA_R_8I -> CUDA_R_32I，内核 gemm_i8 做 int8 乘、int32 累加，比对精确。

DeepBench（baidu-research/DeepBench）本身没有内核，它在 `code/kernels/*.h` 的问题集上调用厂商库（cuDNN / cuBLAS / cuSPARSE / MKL ...）；`gemm-int8.cl` 是按该库的语义为 hwacha-cc 写的 OpenCL 实现。hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `gemm_i8_ct`。

问题规模：inference server / device 集的 6 个形状，缩小 16–64 倍（`gemm-int8_main.c` 的 `shapes[]`，每项注明对应的 DeepBench 原形状与缩放比）。输入由固定种子的伪随机数生成；host 对每个形状先在 Rocket 标量核上跑 C 参考实现，再跑 Hwacha 内核，逐元素比对并打印两侧周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `gemm-int8.s` | hwacha-cc 生成的汇编，入口 `net` |
| `gemm-int8.cl` | 按 DeepBench 所调用库的语义写的 OpenCL 内核 |
| `gemm-int8_main.c` | 裸机 host：问题形状表、输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、percentDiff、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make gemm-int8          # 编译 -> gemm-int8/gemm-int8.riscv
make gemm-int8.spike    # 在 Spike 上运行
make gen-gemm-int8      # 从 .cl 重新生成汇编（需要 hwacha-cc）
```

## Spike 结果

| 形状（DeepBench 原形状与缩放） | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|
| 80x44x128 (inference server 5124x700x2048 /64,/16) | 3,186,144 | 86,143 | 37x |
| 35x44x128 (inference server 35x700x2048 /16) | 1,394,244 | 43,123 | 32x |
| 120x1x80 (inference server 7680x1x2560 /64,/32) | 68,308 | 13,693 | 5x |
| 96x1x80 (inference device 3072x1x1024 /32) | 54,652 | 10,975 | 5x |
| 64x1x76 (inference device 64x1x1216 /16) | 34,652 | 6,999 | 5x |
| 96x47x128 (inference device 3072x1500x128 /32) | 4,083,940 | 103,351 | 40x |

结果：gemm-int8 **PASS**。
