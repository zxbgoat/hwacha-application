# gemm

DeepBench 的 **GEMM** 在 Hwacha 上运行：gemm_bench：cublasSgemm 语义的单精度 GEMM，列主序，问题集里的 NN / TN / NT 三种转置组合各一个内核（gemm_nn / gemm_tn / gemm_nt），每个 C 元素一个 work-item。

DeepBench（baidu-research/DeepBench）本身没有内核，它在 `code/kernels/*.h` 的问题集上调用厂商库（cuDNN / cuBLAS / cuSPARSE / MKL ...）；`gemm.cl` 是按该库的语义为 hwacha-cc 写的 OpenCL 实现。hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `gemm_nn_ct`、`gemm_nt_ct`、`gemm_tn_ct`。

问题规模：training 与 inference 集的 7 个形状，缩小 16–128 倍（`gemm_main.c` 的 `shapes[]`，每项注明对应的 DeepBench 原形状与缩放比）。输入由固定种子的伪随机数生成；host 对每个形状先在 Rocket 标量核上跑 C 参考实现，再跑 Hwacha 内核，逐元素比对并打印两侧周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `gemm.s` | hwacha-cc 生成的汇编，入口 `net` |
| `gemm.cl` | 按 DeepBench 所调用库的语义写的 OpenCL 内核 |
| `gemm_main.c` | 裸机 host：问题形状表、输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、percentDiff、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make gemm          # 编译 -> gemm/gemm.riscv
make gemm.spike    # 在 Spike 上运行
make gen-gemm      # 从 .cl 重新生成汇编（需要 hwacha-cc）
```

## Spike 结果

| 形状（DeepBench 原形状与缩放） | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|
| 110x16x110 NN (training 1760x16x1760 NN /16) | 3,893,392 | 31,688 | 123x |
| 128x32x128 NN (training 2048x32x2048 NN /16) | 10,535,440 | 83,816 | 126x |
| 110x16x110 TN (training 1760x16x1760 TN /16) | 3,893,392 | 31,604 | 123x |
| 110x111x110 NT (training 1760x7133x1760 NT /16,/64) | 27,010,312 | 199,633 | 135x |
| 80x71x55 NN (training 5124x9124x1760 /64,/128,/32) | 6,317,312 | 52,124 | 121x |
| 35x44x128 NN (inference server 35x700x2048 /16) | 3,961,600 | 39,344 | 101x |
| 96x1x80 NN (inference device 3072x1x1024 /32) | 154,784 | 10,040 | 15x |

结果：gemm **PASS**。
