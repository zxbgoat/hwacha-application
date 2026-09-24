# s3d

SHOC level2 `s3d` 在 Hwacha 上运行：**27 个内核（gr_base、ratt..ratt10、rdsmh、ratx、ratxb、ratx2、ratx4、qssa、qssab、qssa2、rdwdot..rdwdot10）按 S3D.cpp 的两阶段顺序启动：22 种组分、206 个反应的化学动力学右端项（组分生成率 WDOT）**。

内核文件（`gr_base.cl`、`qssa.cl`、`qssa2.cl`、`qssab.cl`、`ratt.cl`、`ratt10.cl`、`ratt2.cl`、`ratt3.cl`、`ratt4.cl`、`ratt5.cl`、`ratt6.cl`、`ratt7.cl`、`ratt8.cl`、`ratt9.cl`、`ratx.cl`、`ratx2.cl`、`ratx4.cl`、`ratxb.cl`、`rdsmh.cl`、`rdwdot.cl`、`rdwdot10.cl`、`rdwdot2.cl`、`rdwdot3.cl`、`rdwdot6.cl`、`rdwdot7.cl`、`rdwdot8.cl`、`rdwdot9.cl`、`s3d.cl`）是 SHOC 的原版，未做修改（多文件的基准由一个同名 .cl #include 汇总）；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `gr_base_ct`、`qssa2_kernel_ct`、`qssa_kernel_ct`、`qssab_kernel_ct`、`ratt10_kernel_ct`、`ratt2_kernel_ct`、`ratt3_kernel_ct`、`ratt4_kernel_ct`、`ratt5_kernel_ct`、`ratt6_kernel_ct`、`ratt7_kernel_ct`、`ratt8_kernel_ct`、`ratt9_kernel_ct`、`ratt_kernel_ct`、`ratx2_kernel_ct`、`ratx4_kernel_ct`、`ratx_kernel_ct`、`ratxb_kernel_ct`、`rdsmh_kernel_ct`、`rdwdot10_kernel_ct`、`rdwdot2_kernel_ct`、`rdwdot3_kernel_ct`、`rdwdot6_kernel_ct`、`rdwdot7_kernel_ct`、`rdwdot8_kernel_ct`、`rdwdot9_kernel_ct`、`rdwdot_kernel_ct`，host 按 SHOC host 的启动顺序与工作组形状调用（__local 缓冲区是 host 数组：每个 work-group 独占一个 stripmine）。

问题规模：64 个网格点（N_GP 编进内核，-DN_GP=64）；p 1.0132e6、T 1000、y 按 S3D.cpp（`s3d_main.c`）。输入按 SHOC host 的方式生成（固定种子）；host 先在 Rocket 标量核上跑参考实现，再跑 Hwacha 内核，按 SHOC 的判据比对并打印 PASS/FAIL 与周期数。

说明：SHOC 不校验 S3D；这里把同一批 .cl 编成 C（s3d_ref.c）做参考，全部 6 个输出数组相对误差 < 1e-6；exp10 由 hwacha-cc 新增的展开实现。

## 文件

| 文件 | 内容 |
|---|---|
| `s3d.s` | hwacha-cc 生成的汇编，入口 `net` |
| `gr_base.cl` | SHOC 原版 OpenCL 内核 |
| `qssa.cl` | SHOC 原版 OpenCL 内核 |
| `qssa2.cl` | SHOC 原版 OpenCL 内核 |
| `qssab.cl` | SHOC 原版 OpenCL 内核 |
| `ratt.cl` | SHOC 原版 OpenCL 内核 |
| `ratt10.cl` | SHOC 原版 OpenCL 内核 |
| `ratt2.cl` | SHOC 原版 OpenCL 内核 |
| `ratt3.cl` | SHOC 原版 OpenCL 内核 |
| `ratt4.cl` | SHOC 原版 OpenCL 内核 |
| `ratt5.cl` | SHOC 原版 OpenCL 内核 |
| `ratt6.cl` | SHOC 原版 OpenCL 内核 |
| `ratt7.cl` | SHOC 原版 OpenCL 内核 |
| `ratt8.cl` | SHOC 原版 OpenCL 内核 |
| `ratt9.cl` | SHOC 原版 OpenCL 内核 |
| `ratx.cl` | SHOC 原版 OpenCL 内核 |
| `ratx2.cl` | SHOC 原版 OpenCL 内核 |
| `ratx4.cl` | SHOC 原版 OpenCL 内核 |
| `ratxb.cl` | SHOC 原版 OpenCL 内核 |
| `rdsmh.cl` | SHOC 原版 OpenCL 内核 |
| `rdwdot.cl` | SHOC 原版 OpenCL 内核 |
| `rdwdot10.cl` | SHOC 原版 OpenCL 内核 |
| `rdwdot2.cl` | SHOC 原版 OpenCL 内核 |
| `rdwdot3.cl` | SHOC 原版 OpenCL 内核 |
| `rdwdot6.cl` | SHOC 原版 OpenCL 内核 |
| `rdwdot7.cl` | SHOC 原版 OpenCL 内核 |
| `rdwdot8.cl` | SHOC 原版 OpenCL 内核 |
| `rdwdot9.cl` | SHOC 原版 OpenCL 内核 |
| `s3d.cl` | SHOC 原版 OpenCL 内核（汇总 #include） |
| `s3d_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、__errno 与陷阱桩 |
| `s3d_ref.c` | 标量参考：同一批 .cl 编成 C |
| `s3d_ref.h` | 参考实现的声明 |
| `README.md` | 本文件 |

## 编译与运行

```
make s3d          # 编译 -> s3d/s3d.riscv
make s3d.spike    # 在 Spike 上运行
make gen-s3d      # 从 .cl 重新生成汇编（需要 hwacha-cc）
```

## Spike 结果

| 项 | 周期 |
|---|---|
| s3d scalar | 4,581,336 |
| s3d hwacha-cc | 9,113 |

结果：s3d **PASS**。
