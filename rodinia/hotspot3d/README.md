# hotspot3d

Rodinia `hotspot3D` 的 OpenCL 内核在 Hwacha 上运行：**hotspotOpt1（3-D 热传导 stencil，每个 work-item 负责一根 (x,y) 列并沿 z 扫描；2-D NDRange，如 3D.c）**。

内核文件 `hotspot3d.cl` 是 Rodinia 3.1 的原版，未做修改；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `hotspotOpt1_ct`，host 以 OpenCL host 传给 kernel 的同样参数调用它们。

问题规模：32x32x8，2 步，ping-pong（`hotspot3d_main.c` 中 `NX=32 NY=32 NZ=8 LS=8 ITER=2`）。输入由固定种子的伪随机数生成；host 先在 Rocket 标量核上跑一个参考实现，再跑 Hwacha 内核，逐元素比对并打印两者的周期数。

## 文件

| 文件 | 内容 |
|---|---|
| `hotspot3d.s` | hwacha-cc 生成的汇编，入口 `net` |
| `hotspot3d.cl` | Rodinia 原版 OpenCL 内核 |
| `hotspot3d_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make hotspot3d          # 编译 -> hotspot3d/hotspot3d.riscv
make hotspot3d.spike    # 在 Spike 上运行
make gen-hotspot3d      # 从 .cl 重新生成汇编（clang -> hwacha-cc）
```

## Spike 结果

| 内核 | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |
|---|---|---|---|
| hotspot3d | 825,074 | 3,303 | 250x |

结果：hotspot3d **PASS**。
