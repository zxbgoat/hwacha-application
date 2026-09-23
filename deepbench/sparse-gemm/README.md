# sparse-gemm

DeepBench 的 **稀疏 GEMM** 在 Hwacha 上运行：sparse_bench：cusparseScsrmm，A 为 CSR（稀疏度 0.9 / 0.95，按 DeepBench 的方式由均匀随机数阈值化生成）、B 稠密，alpha = 1/k、beta = 0；内核 csrmm 每个 C 元素一个 work-item，沿行的非零元循环。

DeepBench（baidu-research/DeepBench）本身没有内核，它在 `code/kernels/*.h` 的问题集上调用厂商库（cuDNN / cuBLAS / cuSPARSE / MKL ...）；`sparse-gemm.cl` 是按该库的语义为 hwacha-cc 写的 OpenCL 实现。hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `csrmm_ct`。

问题规模：inference server / device 集的 5 个形状缩小 32–64 倍（`sparse-gemm_main.c` 的 `shapes[]`，每项注明对应的 DeepBench 原形状与缩放比）。输入由固定种子的伪随机数生成；host 对每个形状先在 Rocket 标量核上跑 C 参考实现，再跑 Hwacha 内核，逐元素比对并打印两侧周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `sparse-gemm.s` | hwacha-cc 生成的汇编，入口 `net` |
| `sparse-gemm.cl` | 按 DeepBench 所调用库的语义写的 OpenCL 内核 |
| `sparse-gemm_main.c` | 裸机 host：问题形状表、输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、percentDiff、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make sparse-gemm          # 编译 -> sparse-gemm/sparse-gemm.riscv
make sparse-gemm.spike    # 在 Spike 上运行
make gen-sparse-gemm      # 从 .cl 重新生成汇编（需要 hwacha-cc）
```

## Spike 结果

| 形状（DeepBench 原形状与缩放） | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|
| 240x1x80 nnz 914 (server 7680x1x2560 /32) | 16,460 | 639 | 26x |
| 240x47x80 nnz 950 (server 7680x1500x2560 /32) | 793,476 | 3,339 | 238x |
| 168x4x112 nnz 937 (server 10752x4x3584 /64,/32) | 60,494 | 477 | 127x |
| 240x47x80 nnz 1910 (server 7680x1500x2560 /32, 0.9) | 1,337,172 | 3,339 | 400x |
| 168x47x112 nnz 1885 (device 10752x1500x3584 /64,/32, 0.9) | 1,245,240 | 2,367 | 526x |

结果：sparse-gemm **PASS**。
