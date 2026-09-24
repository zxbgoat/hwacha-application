# raft_large

torchvision `raft_large`（光流）在 Hwacha 上的一次前向，与 PyTorch 比对。文档：https://docs.pytorch.org/vision/stable/models.html#optical-flow

比对内容：最后一轮迭代得到的光流场（1x2xHxW），逐元素比对。

权重随机（固定种子，BatchNorm 给随机的 running 统计量使前向非退化），输入随机；同一组权重同时用于 PyTorch 参考与 Hwacha 构建。

## 本 case 的特殊处理

- 两帧堆叠为一个 2x3xHxW 输入；输入 128x128（特征图 /8 后相关金字塔需要 >=16）；`_iters` 次迭代更新。

## 形状与规模

| | 值 |
|---|---|
| 输入 | 2x3x128x128（98,304 个 float）|
| 输出元素数 | 32,768 |
| 参数量 | 5.3M |
| 权重 blob | 37 MB |
| 光流迭代次数 | 4 |

## 文件

| 文件 | 内容 |
|---|---|
| `raft_large_tv.s` | hwacha-cc 生成的汇编，入口 `net` |
| `raft_large_tv_weights.bin.S` | 权重的 `.incbin` 桩，按符号切分 blob；前半段放 `.weights_lo`、后半段放 `.weights_hi`（`split_weights.py`） |
| `raft_large_tv_weights.bin` | 权重 blob（不入 git，`make gen-raft_large` 按固定种子逐字节重建） |
| `raft_large_tv_check.bin` | 输入与 PyTorch 参考输出，host 用 `.incbin` 内嵌 |
| `tv_main.c` | 通用 host：调用 `net`，比对 max\|diff\|（容差 1e-4 + 1e-2·max\|ref\|）与 argmax（分类输出） |
| `hwlib.s` | 卷积 / 池化库内核 |
| `HWMLIRFLAGS` | 本 case 需要的 hwacha-mlir 映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make raft_large          # 编译 -> raft_large/raft_large_tv.riscv
make raft_large.spike    # 在 Spike 上运行（内存按权重大小自动确定）
make gen-raft_large      # 从 PyTorch 重新生成汇编、权重与参考
```

hwacha-mlir 映射：`--unroll-small=2`。

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 2,270,831,671 |
| max\|diff\| | 7e-06 |
| max\|ref\| | 4.75396 |
| argmax（硬件 / 参考） | 6407 / 6407 |

