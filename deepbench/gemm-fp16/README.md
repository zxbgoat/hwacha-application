# gemm-fp16

DeepBench 的 **GEMM（fp16）** 在 Hwacha 上运行：gemm_bench "half" 精度：cublasGemmEx 16F 输入输出、32F 计算（DeepBench 的 "FP16 inputs / FP32 math"）；内核 gemm_nn / gemm_tn / gemm_nt 用 vlxh + vfcvt.s.h 装载、vfmadd.s 累加、vfcvt.h.s 一次舍入写回；gemm_nn_h 是纯半精度算术（vfmadd.h，每步一次舍入），参考按同样的舍入建模，全部精确。

DeepBench（baidu-research/DeepBench）本身没有内核，它在 `code/kernels/*.h` 的问题集上调用厂商库（cuDNN / cuBLAS / cuSPARSE / MKL ...）；`gemm-fp16.cl` 是按该库的语义为 hwacha-cc 写的 OpenCL 实现。hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `gemm_nn_ct`、`gemm_nn_h_ct`、`gemm_nt_ct`、`gemm_tn_ct`。

问题规模：与 gemm 相同的 6 个形状，另加 2 个纯半精度算术（`gemm-fp16_main.c` 的 `shapes[]`，每项注明对应的 DeepBench 原形状与缩放比）。输入由固定种子的伪随机数生成；host 对每个形状先在 Rocket 标量核上跑 C 参考实现，再跑 Hwacha 内核，逐元素比对并打印两侧周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `gemm-fp16.s` | hwacha-cc 生成的汇编，入口 `net` |
| `gemm-fp16.cl` | 按 DeepBench 所调用库的语义写的 OpenCL 内核 |
| `gemm-fp16_main.c` | 裸机 host：问题形状表、输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、percentDiff、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make gemm-fp16          # 编译 -> gemm-fp16/gemm-fp16.riscv
make gemm-fp16.spike    # 在 Spike 上运行
make gen-gemm-fp16      # 从 .cl 重新生成汇编（需要 hwacha-cc）
```

## Spike 结果

| 形状（DeepBench 原形状与缩放） | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|
| 110x16x110 NN (training 1760x16x1760 NN /16) | 9,977,109 | 31,668 | 315x |
| 128x32x128 NN (training 2048x32x2048 NN /16) | 26,979,875 | 83,760 | 322x |
| 110x16x110 TN (training 1760x16x1760 TN /16) | 10,170,722 | 31,605 | 322x |
| 110x111x110 NT (training 1760x7133x1760 NT /16,/64) | 69,218,378 | 199,637 | 347x |
| 80x71x55 NN (training 5124x9124x1760 /64,/128,/32) | 16,266,959 | 52,042 | 313x |
| 35x44x128 NN (inference server 35x700x2048 /16) | 10,144,651 | 39,322 | 258x |
| 110x16x110 NN half math (training 1760x16x1760 NN /16) | 18,205,253 | 31,595 | 576x |
| 80x71x55 NN half math (training 5124x9124x1760 /64,/128,/32) | 29,542,736 | 51,845 | 570x |

结果：gemm-fp16 **PASS**。
