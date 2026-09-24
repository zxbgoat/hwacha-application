# bfs

SHOC level1 `bfs` 在 Hwacha 上运行：**BFS_kernel_warp（bfs_iiit.cl）：逐层同步 BFS，每个 warp 扫 CHUNK_SZ 个顶点、按 lane 展开邻居，flag 告知 host 是否还有下一层；bfs_uiuc_spill.cl 的 5 个内核也编进了 bfs.s，但未写 host**。

内核文件（`bfs.cl`、`bfs_iiit.cl`、`bfs_uiuc_spill.cl`）是 SHOC 的原版，未做修改（多文件的基准由一个同名 .cl #include 汇总）；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `BFS_kernel_SM_block_ct`、`BFS_kernel_multi_block_ct`、`BFS_kernel_one_block_ct`、`BFS_kernel_warp_ct`、`Frontier_copy_ct`、`Reset_kernel_parameters_ct`，host 按 SHOC host 的启动顺序与工作组形状调用（__local 缓冲区是 host 数组：每个 work-group 独占一个 stripmine）。

问题规模：2048 个顶点的 GenerateSimpleKWayGraph（度 2），源点 0（`bfs_main.c`）。输入按 SHOC host 的方式生成（固定种子）；host 先在 Rocket 标量核上跑参考实现，再跑 Hwacha 内核，按 SHOC 的判据比对并打印 PASS/FAIL 与周期数。

说明：hwacha-cc 新增 get_num_groups(0)。

## 文件

| 文件 | 内容 |
|---|---|
| `bfs.s` | hwacha-cc 生成的汇编，入口 `net` |
| `bfs.cl` | SHOC 原版 OpenCL 内核（汇总 #include） |
| `bfs_iiit.cl` | SHOC 原版 OpenCL 内核 |
| `bfs_uiuc_spill.cl` | SHOC 原版 OpenCL 内核 |
| `bfs_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、__errno 与陷阱桩 |
| `README.md` | 本文件 |

## 编译与运行

```
make bfs          # 编译 -> bfs/bfs.riscv
make bfs.spike    # 在 Spike 上运行
make gen-bfs      # 从 .cl 重新生成汇编（需要 hwacha-cc）
```

## Spike 结果

| 项 | 周期 |
|---|---|
| bfs scalar | 106,484 |
| bfs hwacha-cc | 7,563 |

结果：bfs **PASS**。
