# fpn_dcn

Deformable ConvNets（github.com/msracver/Deformable-ConvNets）的 **fpn_dcn** 在 Hwacha 上的一次前向，与 PyTorch 比对。

模型：FPN：ResNet-101 的 C2-C5 经 1x1 侧向连接 + 自顶向下 + 3x3 平滑得到 P2-P5，P6 为 P5 的 stride-2 池化；共享 RPN 头作用于每一级。

`_dcn` 变体：conv5 阶段（res5a-c）的 3x3 卷积换成可变形卷积（论文的做法）。

比对内容：各级 RPN 头输出拼接成一行，逐元素比对（容差 1e-4 + 1e-2·max\|ref\|）。

权重随机（固定种子，未加载 MXNet 的 .params；BatchNorm 给随机 running 统计量），输入随机，64x64 图像；检测模型的 RoI 固定、RPN 头输出计入比对，proposal 选择 / NMS 因形状数据相关未导出。可变形卷积与 PS-RoI 池化是 `../dcn_ops.py` 中的纯张量实现（torch-mlir 无法 lower torchvision 的自定义算子），与 torchvision.ops 数值一致到 1e-7。

## 形状与规模

| | 值 |
|---|---|
| 输入 | 1x3x64x64（12,288 个 float） |
| 输出元素数 | 5,115 |
| 参数量 | 47.3M |
| 权重 blob | 181 MB |

## 文件

| 文件 | 内容 |
|---|---|
| `fpn_dcn_tv.s` | hwacha-cc 生成的汇编，入口 `net` |
| `fpn_dcn_tv_weights.bin.S` | 权重的 `.incbin` 桩（`split_weights.py` 分成 `.weights_lo` / `.weights_hi` 两段） |
| `fpn_dcn_tv_weights.bin` | 权重 blob（不入 git，`make gen-fpn_dcn` 按固定种子重建） |
| `fpn_dcn_tv_check.bin` | 输入与 PyTorch 参考输出，host 用 `.incbin` 内嵌 |
| `tv_main.c` | 通用 host：调用 `net`，比对 max\|diff\|（分类型输出还比对 argmax） |
| `hwlib.s` | 卷积 / 池化库内核 |
| `README.md` | 本文件 |

## 编译与运行

```
make fpn_dcn          # 编译 -> fpn_dcn/fpn_dcn_tv.riscv
make fpn_dcn.spike    # 在 Spike 上运行
make gen-fpn_dcn      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`默认（最内维为 lane）`。

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 84,600,319 |
| max\|diff\| | 0.014804 |
| max\|ref\| | 20.6421 |
| argmax（硬件 / 参考） | 3930 / 3930 |

