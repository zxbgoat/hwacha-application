# particlefilter

Rodinia `particlefilter (naive)` 的 OpenCL 内核在 Hwacha 上运行：**particle_kernel（重采样：每个粒子线性扫描 CDF 找第一个 >= u[i] 的项并复制该粒子状态；double）**。

内核文件 `particlefilter.cl` 是 Rodinia 3.1 的原版，未做修改；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `particle_kernel_ct`，host 以 OpenCL host 传给 kernel 的同样参数调用它们。

问题规模：512 个粒子，work-group 128（`particlefilter_main.c` 中 `NP=512`）。输入由固定种子的伪随机数生成；host 先在 Rocket 标量核上跑一个参考实现，再跑 Hwacha 内核，逐元素比对并打印两者的周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `particlefilter.s` | hwacha-cc 生成的汇编，入口 `net` |
| `particlefilter.cl` | Rodinia 原版 OpenCL 内核 |
| `particlefilter_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make particlefilter          # 编译 -> particlefilter/particlefilter.riscv
make particlefilter.spike    # 在 Spike 上运行
make gen-particlefilter      # 从 .cl 重新生成汇编（clang -> hwacha-cc）
```

## Spike 结果

| 内核 | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|
| particlefilter | 802,542 | 150 | 5350x |

结果：particlefilter **PASS**。
