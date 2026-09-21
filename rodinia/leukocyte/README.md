# leukocyte

Rodinia `leukocyte（检测阶段）` 的 OpenCL 内核在 Hwacha 上运行：**GICOV_kernel（每个像素在 NCIRCLES 个圆、每圆 NPOINTS 个采样点上计算梯度投影的方差归一化均值的最大值）+ dilate_kernel（strel 窗口内取最大）；buffer 版本**。

内核文件 `leukocyte.cl` 是 Rodinia 3.1 的原版，未做修改；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `GICOV_kernel_ct`, `dilate_kernel_ct`，host 以 OpenCL host 传给 kernel 的同样参数调用它们。

问题规模：24x24 像素（梯度图带 MAX_RAD+2 边界），7 圆 x 150 点，strel 25x25（`leukocyte_main.c` 中 `W=24 H=24 NPOINTS=150 NCIRCLES=7 MAX_RAD=20 STREL=25`）。输入由固定种子的伪随机数生成；host 先在 Rocket 标量核上跑一个参考实现，再跑 Hwacha 内核，逐元素比对并打印两者的周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `leukocyte.s` | hwacha-cc 生成的汇编，入口 `net` |
| `leukocyte.cl` | Rodinia 原版 OpenCL 内核 |
| `leukocyte_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make leukocyte          # 编译 -> leukocyte/leukocyte.riscv
make leukocyte.spike    # 在 Spike 上运行
make gen-leukocyte      # 从 .cl 重新生成汇编（clang -> hwacha-cc）
```

## Spike 结果

| 内核 | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|
| leukocyte | 61,542,998 | 76,685 | 803x |

结果：leukocyte **PASS**。
