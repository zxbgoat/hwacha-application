# cfd

Rodinia `cfd (euler3d)` 的 OpenCL 内核在 Hwacha 上运行：**memset_kernel、initialize_variables、compute_step_factor、compute_flux、time_step：3 阶 RK 的欧拉方程求解，如 euler3d.cpp 的主循环**。

内核文件 `cfd.cl` 是 Rodinia 3.1 的原版，未做修改；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `compute_flux_ct`, `compute_step_factor_ct`, `initialize_variables_ct`, `memset_kernel_ct`, `time_step_ct`，host 以 OpenCL host 传给 kernel 的同样参数调用它们。

问题规模：256 个单元的合成网格（环形邻接 + 随机邻居，含 wing / far-field 面），2 次迭代（`cfd_main.c` 中 `NEL=256 ITER=2`）。输入由固定种子的伪随机数生成；host 先在 Rocket 标量核上跑一个参考实现，再跑 Hwacha 内核，逐元素比对并打印两者的周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `cfd.s` | hwacha-cc 生成的汇编，入口 `net` |
| `cfd.cl` | Rodinia 原版 OpenCL 内核 |
| `cfd_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make cfd          # 编译 -> cfd/cfd.riscv
make cfd.spike    # 在 Spike 上运行
make gen-cfd      # 从 .cl 重新生成汇编（clang -> hwacha-cc）
```

## Spike 结果

| 内核 | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|
| cfd | 976,175 | 10,031 | 97x |

结果：cfd **PASS**。
