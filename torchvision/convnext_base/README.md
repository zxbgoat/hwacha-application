# convnext_base

torchvision `convnext_base`（图像分类）在 Hwacha 上的一次前向，与 PyTorch 比对。文档：https://docs.pytorch.org/vision/stable/models.html#classification

比对内容：1000 类 logits，逐元素比对并要求 argmax 一致。

权重随机（固定种子，BatchNorm 给随机的 running 统计量使前向非退化），输入随机；同一组权重同时用于 PyTorch 参考与 Hwacha 构建。

## 形状与规模

| | 值 |
|---|---|
| 输入 | 1x3x32x32（3,072 个 float）|
| 输出元素数 | 1,000 |
| 参数量 | 88.6M |
| 权重 blob | 338 MB |

## 文件

| 文件 | 内容 |
|---|---|
| `convnext_base_tv.s` | hwacha-cc 生成的汇编，入口 `net` |
| `convnext_base_tv_weights.bin.S` | 权重的 `.incbin` 桩，按符号切分 blob；前半段放 `.weights_lo`、后半段放 `.weights_hi`（`split_weights.py`） |
| `convnext_base_tv_weights.bin` | 权重 blob（不入 git，`make gen-convnext_base` 按固定种子逐字节重建） |
| `convnext_base_tv_check.bin` | 输入与 PyTorch 参考输出，host 用 `.incbin` 内嵌 |
| `tv_main.c` | 通用 host：调用 `net`，比对 max\|diff\|（容差 1e-4 + 1e-2·max\|ref\|）与 argmax（分类输出） |
| `hwlib.s` | 卷积 / 池化库内核 |
| `HWMLIRFLAGS` | 本 case 需要的 hwacha-mlir 映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make convnext_base          # 编译 -> convnext_base/convnext_base_tv.riscv
make convnext_base.spike    # 在 Spike 上运行（内存按权重大小自动确定）
make gen-convnext_base      # 从 PyTorch 重新生成汇编、权重与参考
```

hwacha-mlir 映射：`--collapse-all`。

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 12,510,423 |
| max\|diff\| | 3e-06 |
| max\|ref\| | 2.05513 |
| argmax（硬件 / 参考） | 864 / 864 |

