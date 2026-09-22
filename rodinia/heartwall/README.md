# heartwall

Rodinia `heartwall` 的 OpenCL 内核在 Hwacha 上运行：**kernel_gpu_opencl：每个 work-group（64 个 lane）跟踪超声心动视频中的一个采样点，第 0 帧提取模板，之后每帧在搜索窗内做归一化互相关（累积和实现）并施加位移掩码；hwacha-cc 以 `-vregs 32` 编译**。

内核文件 `heartwall.cl` 是 Rodinia 3.1 的原版，未做修改；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `kernel_gpu_opencl_ct`，host 以 OpenCL host 传给 kernel 的同样参数调用它们。

问题规模：51 个点（20 心内膜 + 31 心外膜），3 帧 560x480 合成纹理（每帧平移 1 行 1 列），tSize 5 / sSize 8（`heartwall_main.c` 中 `FRAMES=3 ROWS=560 COLS=480 T_SIZE=5 S_SIZE=8`）。输入由固定种子的伪随机数生成；host 先在 Rocket 标量核上跑一个参考实现，再跑 Hwacha 内核，逐元素比对并打印两者的周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `heartwall.s` | hwacha-cc 生成的汇编，入口 `net` |
| `heartwall.cl` | Rodinia 原版 OpenCL 内核 |
| `heartwall_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、__errno 与陷阱桩 |
| `heartwall_ref.c` | Rodinia OpenMP 版的 kernel.c + define.c（仅改函数名）作为标量参考 |
| `heartwall_ref.h` | 参考实现的 public_struct / private_struct 与入口声明 |
| `main.h` | Rodinia OpenCL 版的 main.h（params_common、NUMBER_THREADS = RD_WG_SIZE） |
| `README.md` | 本文件 |

## 编译与运行

```
make heartwall          # 编译 -> heartwall/heartwall.riscv
make heartwall.spike    # 在 Spike 上运行
make gen-heartwall      # 从 .cl 重新生成汇编（clang -> hwacha-cc）
```

## Spike 结果

| 内核 | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|
| heartwall | 1,991,790 | 5,773 | 345x |

结果：heartwall **PASS**。
