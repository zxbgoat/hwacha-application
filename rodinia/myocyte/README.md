# myocyte

Rodinia `myocyte` 的 OpenCL 内核在 Hwacha 上运行：**kernel_gpu_opencl（group 0 / lane 0 跑 ECC 模型，group 1 / lane 0 跑三次 CaM 模型：一次 ODE 右端项求值；两个被调函数标为 always_inline）**。

内核文件 `myocyte.cl` 是 Rodinia 3.1 的原版，唯一改动：kernel_ecc / kernel_cam 加了 __attribute__((always_inline))；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `kernel_gpu_opencl_ct`，host 以 OpenCL host 传给 kernel 的同样参数调用它们。

问题规模：91 个方程，18 个参数（`myocyte_main.c` 中 `EQUATIONS=91 PARAMETERS=18`）。输入由固定种子的伪随机数生成；host 先在 Rocket 标量核上跑一个参考实现，再跑 Hwacha 内核，逐元素比对并打印两者的周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `myocyte.s` | hwacha-cc 生成的汇编，入口 `net` |
| `myocyte.cl` | Rodinia 原版 OpenCL 内核（两个辅助函数加了 always_inline） |
| `myocyte_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、__errno 与陷阱桩 |
| `myocyte_ref.c` | 标量参考：同一 .cl 以 C 编译（OpenCL 限定符定义为空） |
| `README.md` | 本文件 |

## 编译与运行

```
make myocyte          # 编译 -> myocyte/myocyte.riscv
make myocyte.spike    # 在 Spike 上运行
make gen-myocyte      # 从 .cl 重新生成汇编（clang -> hwacha-cc）
```

## Spike 结果

| 内核 | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|
| myocyte | 13,591 | 123 | 110x |

结果：myocyte **PASS**。
