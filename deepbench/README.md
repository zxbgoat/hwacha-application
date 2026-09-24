# deepbench

DeepBench（baidu-research/DeepBench）的算子在 Hwacha 上运行，含单精度、int8 推理与半精度。DeepBench 本身没有内核：它在
`code/kernels/*.h` 列出的问题集（GEMM、卷积、RNN、稀疏 GEMM、all-reduce 的形状）上调用厂商库
（cuDNN / cuBLAS / cuSPARSE、MIOpen、MKL、Arm Compute Library）计时。这里每个算子一个 case：`.cl` 是按
对应库的语义为 hwacha-cc 写的 OpenCL 实现，host 跑问题集中缩小后的若干形状，逐个与 Rocket 标量核上的
C 参考实现比对并打印两侧周期数。

| case | DeepBench 基准 | 内核 | 形状（缩小后） | Hwacha |
|---|---|---|---|---|
| gemm | gemm_bench（cublasSgemm，NN / TN / NT） | `gemm_nn`, `gemm_tn`, `gemm_nt` | training + inference 集的 7 个，m,n,k ≤ 128 | PASS（精确） |
| gemm-int8 | gemm_bench inference int8（cublasGemmEx 8I -> 32I） | `gemm_i8` | inference server / device 集的 6 个 | PASS（精确） |
| conv | conv_bench（cudnnConvolutionForward / BackwardData / BackwardFilter，NCHW） | `conv_fwd`, `conv_bwd_data`, `conv_bwd_filter` | training 集的 VGG / ResNet / Inception / DeepSpeech 5 层 | PASS（精确） |
| rnn-vanilla | rnn_bench "vanilla"（cuDNN RNN_RELU，skip-input，单层） | `rnn_relu_step`，每时间步一次启动 | training 集 3 个 | PASS（精确） |
| rnn-lstm | rnn_bench "lstm"（cuDNN LSTM，skip-input，单层） | `lstm_step` | training 集 3 个 | PASS（1e-3） |
| rnn-gru | rnn_bench "gru"（cuDNN GRU，skip-input，单层） | `gru_step` | training 集 3 个 | PASS（1e-3） |
| sparse-gemm | sparse_bench（cusparseScsrmm，稀疏度 0.9 / 0.95，alpha = 1/k） | `csrmm` | inference 集 5 个 | PASS（精确） |
| gemm-fp16 | gemm_bench "half"（GemmEx 16F 输入输出、32F 计算） | `gemm_nn/tn/nt`（vlxh + vfcvt.s.h、vfmadd.s、vfcvt.h.s），`gemm_nn_h`（纯 vfmadd.h） | 与 gemm 相同的 6 个 + 2 个纯半精度 | PASS（精确） |
| conv-fp16 | conv_bench "half"（CUDNN_DATA_HALF 张量，float 计算） | `conv_fwd`, `conv_bwd_data`, `conv_bwd_filter` | 与 conv 相同的 5 层 | PASS（精确） |
| rnn-lstm-fp16 | rnn_bench "lstm" "half"（half 张量与状态，float 门运算） | `lstm_step` | 与 rnn-lstm 相同的 3 个 | PASS（4e-3） |
| rnn-vanilla-fp16 | rnn_bench "vanilla" "half"（half 张量与状态，float 运算） | `rnn_relu_step` | 与 rnn-vanilla 相同的 3 个 | PASS（精确） |
| rnn-gru-fp16 | rnn_bench "gru" "half"（half 张量与状态，float 门运算） | `gru_step` | 与 rnn-gru 相同的 3 个 | PASS（4e-3） |

未迁移：**all-reduce**（`all_reduce_problems.h`，NCCL / MPI 的多设备归约，单个 Hwacha 上没有对应物）；RNN
只有前向（DeepBench 的 inference 模式；training 模式还计 cudnnRNNBackwardData / Weights）。fp16 覆盖 GEMM、
卷积和三种 RNN。

**半精度**：Hwacha 有完整的 half 指令（`vfadd.h` / `vfmadd.h` / `vfcvt.s.h` / `vfcvt.h.s`、混合精度的
`vfmadd.s.h`），hwacha-cc 对 OpenCL `half`（`cl_khr_fp16`，Makefile 的 CLFLAGS 已开启）生成 `vlxh` +
`vfcvt.s.h` 装载、`vfcvt.h.s` + `vsxh` 存回，纯 half 表达式生成 `vfmadd.h`。host 侧 gcc 9.2 没有
`_Float16`，`common.h` 提供 `f16` 类型与 RNE 的 `f32_to_f16` / `f16_to_f32` / `h_round`。Spike 的 Hwacha
模型里所有 half 结果的写回（`WRITE_HFRD`）原本用标量编码的舍入模式字段，等于截断——hwacha-compiler 的
`patches/esp-isa-sim-hwacha-half-rm.patch` 改为向量指令自己的 `VFRM`，三个 fp16 case 需要打了这个补丁的
Spike（在此之前"半精度输入、单精度累加"仍精确，纯 half 算术与 `vfcvt.h.s` 的结果偏小）。

```
make            # 编译全部 -> <case>/<case>.riscv
make run        # 全部在 Spike 上运行，每个 case 一行 PASS/FAIL
make <case>     # 编译一个
make <case>.spike
make gen-<case> # 从 .cl 重新生成汇编（需要 hwacha-cc）
```

RNN 的 sigmoid / tanh 用 `exp` 写成，hwacha-cc 用多项式展开 `exp`，与 libm 的 `expf` 相比在 1e-3 相对误差
内（实测 `max|diff| < 1e-3`）；其余 case 与参考逐元素精确相等。int8 case 的 host 用 `int8_t`：RISC-V 上
C 的 `char` 无符号，OpenCL 的 `char` 有符号。周期数见各 case 目录的 README（Spike 的 rdcycle 只计标量核
指令，向量指令不占周期，加速比因此偏大——各套件一样）。
