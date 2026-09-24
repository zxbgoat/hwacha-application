# md5hash

SHOC level1 `md5hash` 在 Hwacha 上运行：**FindKeyWithDigest_Kernel：在密钥空间里暴力搜索 MD5 摘要等于目标的密钥，每个 work-item 处理 valsPerByte 个连续密钥**。

内核文件（`md5hash.cl`）是 SHOC 的原版，未做修改（多文件的基准由一个同名 .cl #include 汇总）；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `FindKeyWithDigest_Kernel_ct`，host 按 SHOC host 的启动顺序与工作组形状调用（__local 缓冲区是 host 数组：每个 work-group 独占一个 stripmine）。

问题规模：4 字节 x 10 个取值 = 10^4 个密钥（SHOC 7 x 10），3 个随机目标（`md5hash_main.c`）。输入按 SHOC host 的方式生成（固定种子）；host 先在 Rocket 标量核上跑参考实现，再跑 Hwacha 内核，按 SHOC 的判据比对并打印 PASS/FAIL 与周期数。

说明：clang 以 -Dinline=static 编译保留 md5_2words 的定义；LEFTROTATE 形成的 llvm.fshl 由 hwacha-cc 展开。

## 文件

| 文件 | 内容 |
|---|---|
| `md5hash.s` | hwacha-cc 生成的汇编，入口 `net` |
| `md5hash.cl` | SHOC 原版 OpenCL 内核 |
| `md5hash_main.c` | 裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数 |
| `common.h` | 伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、__errno 与陷阱桩 |
| `md5hash_ref.c` | 标量参考：SHOC host 侧的 md5_2words / IndexToKey |
| `md5hash_ref.h` | 参考实现的声明 |
| `README.md` | 本文件 |

## 编译与运行

```
make md5hash          # 编译 -> md5hash/md5hash.riscv
make md5hash.spike    # 在 Spike 上运行
make gen-md5hash      # 从 .cl 重新生成汇编（需要 hwacha-cc）
```

## Spike 结果

| 项 | 周期 |
|---|---|
| md5hash pass 0 / digest matches | 449 |
| md5hash pass 1 / digest matches | 449 |
| md5hash pass 2 / digest matches | 449 |

结果：md5hash **PASS**。
