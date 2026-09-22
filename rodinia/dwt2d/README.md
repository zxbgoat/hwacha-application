# dwt2d

Rodinia `dwt2d` 的 OpenCL 内核在 Hwacha 上运行：**cl_fdwt53Kernel：一级正向 5/3 整数提升小波变换，每个 work-group 处理一个 32x8 的滑动窗口（先列后行，四个象限带输出）；hwacha-cc 以 `-vregs 64` 编译使 32 个 lane 的组落在一个 stripmine 内**。

内核文件 `dwt2d.cl` 是 Rodinia 3.1 的原版，未做修改；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `c_CopySrcToComponent_ct`, `c_CopySrcToComponents_ct`, `cl_fdwt53Kernel_ct`，host 以 OpenCL host 传给 kernel 的同样参数调用它们。

问题规模：64x64 整数图像（`dwt2d_main.c` 中 `SX=64 SY=64 WIN_SX=32 WIN_SY=8`）。输入由固定种子的伪随机数生成；host 先在 Rocket 标量核上跑一个参考实现，再跑 Hwacha 内核，逐元素比对并打印两者的周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `dwt2d.s` | hwacha-cc 生成的汇编，入口 `net` |
| `dwt2d.cl` | Rodinia 原版 OpenCL 内核 |
| `dwt2d_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make dwt2d          # 编译 -> dwt2d/dwt2d.riscv
make dwt2d.spike    # 在 Spike 上运行
make gen-dwt2d      # 从 .cl 重新生成汇编（clang -> hwacha-cc）
```

## Spike 结果

| 内核 | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|
| dwt2d | 262,417 | 459 | 572x |

结果：dwt2d **PASS**。
