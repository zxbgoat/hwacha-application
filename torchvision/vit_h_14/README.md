# vit_h_14

torchvision `vit_h_14`（图像分类）在 Hwacha 上的一次前向，与 PyTorch 比对。文档：https://docs.pytorch.org/vision/stable/models.html#classification

比对内容：1000 类 logits，逐元素比对并要求 argmax 一致。

权重随机（固定种子，BatchNorm 给随机的 running 统计量使前向非退化），输入随机；同一组权重同时用于 PyTorch 参考与 Hwacha 构建。

## 本 case 的特殊处理

- `image_size` 设为输入尺寸；torchvision 把 ViT 的分类头零初始化（logits 恒为 0），导出时给全零 Linear 一个小随机初始化。
- 权重 2.4 GB，超过 PC 相对寻址范围：`split_weights.py` 把权重 blob 对半分到代码前后两个段（`tv.ld`），模拟器内存按权重大小自动给到 4 GB。

## 形状与规模

| | 值 |
|---|---|
| 输入 | 1x3x56x56（9,408 个 float）|
| 输出元素数 | 1,000 |
| 参数量 | 632.0M |
| 权重 blob | 2410 MB |
| 构造参数 | `{'image_size': 56}` |

## 文件

| 文件 | 内容 |
|---|---|
| `vit_h_14_tv.s` | hwacha-cc 生成的汇编，入口 `net` |
| `vit_h_14_tv_weights.bin.S` | 权重的 `.incbin` 桩，按符号切分 blob；前半段放 `.weights_lo`、后半段放 `.weights_hi`（`split_weights.py`） |
| `vit_h_14_tv_weights.bin` | 权重 blob（不入 git，`make gen-vit_h_14` 按固定种子逐字节重建） |
| `vit_h_14_tv_check.bin` | 输入与 PyTorch 参考输出，host 用 `.incbin` 内嵌 |
| `tv_main.c` | 通用 host：调用 `net`，比对 max\|diff\|（容差 1e-4 + 1e-2·max\|ref\|）与 argmax（分类输出） |
| `hwlib.s` | 卷积 / 池化库内核 |
| `README.md` | 本文件 |

## 编译与运行

```
make vit_h_14          # 编译 -> vit_h_14/vit_h_14_tv.riscv
make vit_h_14.spike    # 在 Spike 上运行（内存按权重大小自动确定）
make gen-vit_h_14      # 从 PyTorch 重新生成汇编、权重与参考
```

hwacha-mlir 映射：`默认（最内维为 lane）`。

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 132,091,655 |
| max\|diff\| | 6e-06 |
| max\|ref\| | 2.49015 |
| argmax（硬件 / 参考） | 199 / 199 |

