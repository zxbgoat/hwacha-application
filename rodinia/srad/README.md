# srad

Rodinia `srad` 的 OpenCL 内核在 Hwacha 上运行：**extract、prepare + reduce（均值 / 方差）、srad_kernel、srad2_kernel、compress，如 kernel_gpu_opencl_wrapper.c**。

内核文件 `srad.cl` 是 Rodinia 3.1 的原版，未做修改；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `compress_kernel_ct`, `extract_kernel_ct`, `prepare_kernel_ct`, `reduce_kernel_ct`, `srad2_kernel_ct`, `srad_kernel_ct`，host 以 OpenCL host 传给 kernel 的同样参数调用它们。

问题规模：32x32 图像，2 次迭代；NUMBER_THREADS 64（`srad_main.c` 中 `NR=32 NC=32 NITER=2`）。输入由固定种子的伪随机数生成；host 先在 Rocket 标量核上跑一个参考实现，再跑 Hwacha 内核，逐元素比对并打印两者的周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `srad.s` | hwacha-cc 生成的汇编，入口 `net` |
| `srad.cl` | Rodinia 原版 OpenCL 内核 |
| `srad_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、__errno 与陷阱桩 |
| `main.h` | 内核 #include 的 Rodinia host 头文件的替身（fp、NUMBER_THREADS） |
| `README.md` | 本文件 |

## 编译与运行

```
make srad          # 编译 -> srad/srad.riscv
make srad.spike    # 在 Spike 上运行
make gen-srad      # 从 .cl 重新生成汇编（clang -> hwacha-cc）
```

## Spike 结果

| 内核 | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|


## 已知问题

srad_kernel 的最终 store 用一个已被复用的寄存器做索引（hwacha-cc 寄存器分配问题），Spike 上 STORE ACCESS FAULT；见 `../known-issues/README.md`。
