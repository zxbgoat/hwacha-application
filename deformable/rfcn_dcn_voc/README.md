# rfcn_dcn_voc

Deformable ConvNets（github.com/msracver/Deformable-ConvNets）的 **rfcn_dcn_voc** 在 Hwacha 上的一次前向，与 PyTorch 比对。

模型：R-FCN：ResNet-101 conv5 空洞化 -> 1x1 降维到 1024 -> 位置敏感的类别得分图（k*k*C）与 bbox 图（k*k*4）-> 3 个固定 RoI 的 PS-RoI 池化 -> 各 bin 平均投票；conv5 上另有 RPN 头（VOC 21 类）。

`_dcn` 变体：conv5 阶段（res5a-c）的 3x3 卷积换成可变形卷积（论文的做法），RoI 池化换成可变形 PS-RoI 池化（每个 bin 的偏移由第一次池化的特征经 3x3 卷积预测，gamma=0.1）。

比对内容：RPN 的 cls/box 图、类别投票、bbox 投票拼接成一行，逐元素比对（容差 1e-4 + 1e-2·max\|ref\|）。

权重随机（固定种子，未加载 MXNet 的 .params；BatchNorm 给随机 running 统计量），输入随机，64x64 图像；检测模型的 RoI 固定、RPN 头输出计入比对，proposal 选择 / NMS 因形状数据相关未导出。可变形卷积与 PS-RoI 池化是 `../dcn_ops.py` 中的纯张量实现（torch-mlir 无法 lower torchvision 的自定义算子），与 torchvision.ops 数值一致到 1e-7。

## 形状与规模

| | 值 |
|---|---|
| 输入 | 1x3x64x64（12,288 个 float） |
| 输出元素数 | 795 |
| 参数量 | 55.6M |
| 权重 blob | 212 MB |

## 文件

| 文件 | 内容 |
|---|---|
| `rfcn_dcn_voc_tv.s` | hwacha-cc 生成的汇编，入口 `net` |
| `rfcn_dcn_voc_tv_weights.bin.S` | 权重的 `.incbin` 桩（`split_weights.py` 分成 `.weights_lo` / `.weights_hi` 两段） |
| `rfcn_dcn_voc_tv_weights.bin` | 权重 blob（不入 git，`make gen-rfcn_dcn_voc` 按固定种子重建） |
| `rfcn_dcn_voc_tv_check.bin` | 输入与 PyTorch 参考输出，host 用 `.incbin` 内嵌 |
| `tv_main.c` | 通用 host：调用 `net`，比对 max\|diff\|（分类型输出还比对 argmax） |
| `hwlib.s` | 卷积 / 池化库内核 |
| `HWMLIRFLAGS` | 本 case 需要的 hwacha-mlir 映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make rfcn_dcn_voc          # 编译 -> rfcn_dcn_voc/rfcn_dcn_voc_tv.riscv
make rfcn_dcn_voc.spike    # 在 Spike 上运行
make gen-rfcn_dcn_voc      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--unroll-small=2`。

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 120,045,888 |
| max\|diff\| | 0.017336 |
| max\|ref\| | 23.852 |
| argmax（硬件 / 参考） | 633 / 633 |

