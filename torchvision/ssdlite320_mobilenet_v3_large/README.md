# ssdlite320_mobilenet_v3_large

torchvision `ssdlite320_mobilenet_v3_large`（目标检测）在 Hwacha 上的一次前向，与 PyTorch 比对。文档：https://docs.pytorch.org/vision/stable/models.html#object-detection

比对内容：网络部分的输出：骨干 + FPN + 检测头在全部 anchor 上的输出拼成一行（两阶段模型到 RPN 头为止），逐元素比对。后处理（分数阈值、NMS）的输出形状数据相关，torch.export 无法静态化，未包含。

权重随机（固定种子，BatchNorm 给随机的 running 统计量使前向非退化），输入随机；同一组权重同时用于 PyTorch 参考与 Hwacha 构建。

## 形状与规模

| | 值 |
|---|---|
| 输入 | 1x3x64x64（12,288 个 float）|
| 输出元素数 | 13,680 |
| 参数量 | 3.4M |
| 权重 blob | 13 MB |

## 文件

| 文件 | 内容 |
|---|---|
| `ssdlite320_mobilenet_v3_large_tv.s` | hwacha-cc 生成的汇编，入口 `net` |
| `ssdlite320_mobilenet_v3_large_tv_weights.bin.S` | 权重的 `.incbin` 桩，按符号切分 blob；前半段放 `.weights_lo`、后半段放 `.weights_hi`（`split_weights.py`） |
| `ssdlite320_mobilenet_v3_large_tv_weights.bin` | 权重 blob（不入 git，`make gen-ssdlite320_mobilenet_v3_large` 按固定种子逐字节重建） |
| `ssdlite320_mobilenet_v3_large_tv_check.bin` | 输入与 PyTorch 参考输出，host 用 `.incbin` 内嵌 |
| `tv_main.c` | 通用 host：调用 `net`，比对 max\|diff\|（容差 1e-4 + 1e-2·max\|ref\|）与 argmax（分类输出） |
| `hwlib.s` | 卷积 / 池化库内核 |
| `HWMLIRFLAGS` | 本 case 需要的 hwacha-mlir 映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make ssdlite320_mobilenet_v3_large          # 编译 -> ssdlite320_mobilenet_v3_large/ssdlite320_mobilenet_v3_large_tv.riscv
make ssdlite320_mobilenet_v3_large.spike    # 在 Spike 上运行（内存按权重大小自动确定）
make gen-ssdlite320_mobilenet_v3_large      # 从 PyTorch 重新生成汇编、权重与参考
```

hwacha-mlir 映射：`--collapse-all`。

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 12,880,551 |
| max\|diff\| | 0 |
| max\|ref\| | 0.266549 |
| argmax（硬件 / 参考） | 7659 / 7659 |

