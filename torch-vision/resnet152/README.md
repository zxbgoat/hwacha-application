# resnet152

torchvision `resnet152`（图像分类）在 Hwacha 上的一次前向，与 PyTorch 比对。文档：https://docs.pytorch.org/vision/stable/models.html#classification

比对内容：1000 类 logits，逐元素比对并要求 argmax 一致。

权重随机（固定种子，BatchNorm 给随机的 running 统计量使前向非退化），输入随机；同一组权重同时用于 PyTorch 参考与 Hwacha 构建。

## 形状与规模

| | 值 |
|---|---|
| 输入 | 1x3x32x32（3,072 个 float）|
| 输出元素数 | 1,000 |
| 参数量 | 60.2M |
| 权重 blob | 230 MB |

## 文件

| 文件 | 内容 |
|---|---|
| `resnet152_tv.s` | hwacha-cc 生成的汇编，入口 `net` |
| `resnet152_tv_weights.bin.S` | 权重的 `.incbin` 桩，按符号切分 blob；前半段放 `.weights_lo`、后半段放 `.weights_hi`（`split_weights.py`） |
| `resnet152_tv_weights.bin` | 权重 blob（不入 git，`make gen-resnet152` 按固定种子逐字节重建） |
| `resnet152_tv_check.bin` | 输入与 PyTorch 参考输出，host 用 `.incbin` 内嵌 |
| `tv_main.c` | 通用 host：调用 `net`，比对 max\|diff\|（容差 1e-4 + 1e-2·max\|ref\|）与 argmax（分类输出） |
| `hwlib.s` | 卷积 / 池化库内核 |
| `README.md` | 本文件 |

## 编译与运行

```
make resnet152          # 编译 -> resnet152/resnet152_tv.riscv
make resnet152.spike    # 在 Spike 上运行（内存按权重大小自动确定）
make gen-resnet152      # 从 PyTorch 重新生成汇编、权重与参考
```

hwacha-mlir 映射：`默认（最内维为 lane）`。

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 103,648,035 |
| max\|diff\| | 31 |
| max\|ref\| | 2.12601e+07 |
| argmax（硬件 / 参考） | 704 / 704 |

