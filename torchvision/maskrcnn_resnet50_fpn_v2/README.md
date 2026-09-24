# maskrcnn_resnet50_fpn_v2

torchvision `maskrcnn_resnet50_fpn_v2`（目标检测）在 Hwacha 上的一次前向，与 PyTorch 比对。文档：https://docs.pytorch.org/vision/stable/models.html#object-detection

比对内容：网络部分的输出：骨干 + FPN + 检测头在全部 anchor 上的输出拼成一行（两阶段模型到 RPN 头为止），逐元素比对。后处理（分数阈值、NMS）的输出形状数据相关，torch.export 无法静态化，未包含。

权重随机（固定种子，BatchNorm 给随机的 running 统计量使前向非退化），输入随机；同一组权重同时用于 PyTorch 参考与 Hwacha 构建。

## 形状与规模

| | 值 |
|---|---|
| 输入 | 1x3x64x64（12,288 个 float）|
| 输出元素数 | 5,115 |
| 参数量 | 46.4M |
| 权重 blob | 107 MB |

## 文件

| 文件 | 内容 |
|---|---|
| `maskrcnn_resnet50_fpn_v2_tv.s` | hwacha-cc 生成的汇编，入口 `net` |
| `maskrcnn_resnet50_fpn_v2_tv_weights.bin.S` | 权重的 `.incbin` 桩，按符号切分 blob；前半段放 `.weights_lo`、后半段放 `.weights_hi`（`split_weights.py`） |
| `maskrcnn_resnet50_fpn_v2_tv_weights.bin` | 权重 blob（不入 git，`make gen-maskrcnn_resnet50_fpn_v2` 按固定种子逐字节重建） |
| `maskrcnn_resnet50_fpn_v2_tv_check.bin` | 输入与 PyTorch 参考输出，host 用 `.incbin` 内嵌 |
| `tv_main.c` | 通用 host：调用 `net`，比对 max\|diff\|（容差 1e-4 + 1e-2·max\|ref\|）与 argmax（分类输出） |
| `hwlib.s` | 卷积 / 池化库内核 |
| `README.md` | 本文件 |

## 编译与运行

```
make maskrcnn_resnet50_fpn_v2          # 编译 -> maskrcnn_resnet50_fpn_v2/maskrcnn_resnet50_fpn_v2_tv.riscv
make maskrcnn_resnet50_fpn_v2.spike    # 在 Spike 上运行（内存按权重大小自动确定）
make gen-maskrcnn_resnet50_fpn_v2      # 从 PyTorch 重新生成汇编、权重与参考
```

hwacha-mlir 映射：`默认（最内维为 lane）`。

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 45,794,495 |
| max\|diff\| | 7.8e-05 |
| max\|ref\| | 21.669 |
| argmax（硬件 / 参考） | 3214 / 3214 |

