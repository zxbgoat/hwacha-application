# fft

SHOC level1 `fft` 在 Hwacha 上运行：**fft1D_512 / ifft1D_512（64 个 work-item 一组做一个 512 点复数 FFT：基 8 三遍、旋转因子、两次经 __local 的转置）+ chk1D_512（SHOC 的自检内核）**。

内核文件（`fft.cl`）是 SHOC 的原版，未做修改（多文件的基准由一个同名 .cl #include 汇总）；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `chk1D_512_ct`、`fft1D_512_ct`、`ifft1D_512_ct`，host 按 SHOC host 的启动顺序与工作组形状调用（__local 缓冲区是 host 数组：每个 work-group 独占一个 stripmine）。

问题规模：8 个 512 点 FFT（SHOC 256 个），后半批是前半批的副本（`fft_main.c`）。输入按 SHOC host 的方式生成（固定种子）；host 先在 Rocket 标量核上跑参考实现，再跑 Hwacha 内核，按 SHOC 的判据比对并打印 PASS/FAIL 与周期数。

说明：float2 由 scalarizer 拆开；旋转因子的 sin / cos 由 hwacha-cc 新增的展开实现。

## 文件

| 文件 | 内容 |
|---|---|
| `fft.s` | hwacha-cc 生成的汇编，入口 `net` |
| `fft.cl` | SHOC 原版 OpenCL 内核 |
| `fft_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make fft          # 编译 -> fft/fft.riscv
make fft.spike    # 在 Spike 上运行
make gen-fft      # 从 .cl 重新生成汇编（需要 hwacha-cc）
```

## Spike 结果

| 项 | 周期 |
|---|---|
| fft scalar /  | 266,426,419 |
| fft hwacha-cc forward | 10,727 |
| 0 mismatches fft hwacha-cc inverse | 11,468 |

结果：fft **PASS**。
