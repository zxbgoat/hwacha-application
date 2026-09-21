# deform_conv_v2

Deformable ConvNets（github.com/msracver/Deformable-ConvNets）的 **deform_conv_v2** 在 Hwacha 上的一次前向，与 PyTorch 比对。

模型：一个 3x3 可变形卷积（DCN v2）：偏移量加 sigmoid 调制掩码。

比对内容：输出特征图 1x64x16x16，逐元素比对（容差 1e-4 + 1e-2·max\|ref\|）。

权重随机（固定种子，未加载 MXNet 的 .params；BatchNorm 给随机 running 统计量），输入随机，64x64 图像；检测模型的 RoI 固定、RPN 头输出计入比对，proposal 选择 / NMS 因形状数据相关未导出。可变形卷积与 PS-RoI 池化是 `../dcn_ops.py` 中的纯张量实现（torch-mlir 无法 lower torchvision 的自定义算子），与 torchvision.ops 数值一致到 1e-7。

## 形状与规模

| | 值 |
|---|---|
| 输入 | 1x64x16x16（16,384 个 float） |
| 输出元素数 | 16,384 |
| 参数量 | 0.1M |
| 权重 blob | 0 MB |

## 文件

| 文件 | 内容 |
|---|---|
| `deform_conv_v2_tv.s` | hwacha-cc 生成的汇编，入口 `net` |
| `deform_conv_v2_tv_weights.bin.S` | 权重的 `.incbin` 桩（`split_weights.py` 分成 `.weights_lo` / `.weights_hi` 两段） |
| `deform_conv_v2_tv_weights.bin` | 权重 blob（不入 git，`make gen-deform_conv_v2` 按固定种子重建） |
| `deform_conv_v2_tv_check.bin` | 输入与 PyTorch 参考输出，host 用 `.incbin` 内嵌 |
| `tv_main.c` | 通用 host：调用 `net`，比对 max\|diff\|（分类型输出还比对 argmax） |
| `hwlib.s` | 卷积 / 池化库内核 |
| `README.md` | 本文件 |

## 编译与运行

```
make deform_conv_v2          # 编译 -> deform_conv_v2/deform_conv_v2_tv.riscv
make deform_conv_v2.spike    # 在 Spike 上运行
make gen-deform_conv_v2      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`默认（最内维为 lane）`。

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 1,106,067 |
| max\|diff\| | 3e-06 |
| max\|ref\| | 2.29301 |
| argmax（硬件 / 参考） | 1060 / 1060 |

