# deform_psroi

Deformable ConvNets（github.com/msracver/Deformable-ConvNets）的 **deform_psroi** 在 Hwacha 上的一次前向，与 PyTorch 比对。

模型：可变形位置敏感 RoI 池化：先做一次常规 PS-RoI 池化，3x3 卷积预测每个 bin 的 (dy, dx)，再按偏移双线性采样。

比对内容：池化结果 (3, 4, 7, 7)，逐元素比对（容差 1e-4 + 1e-2·max\|ref\|）。

权重随机（固定种子，未加载 MXNet 的 .params；BatchNorm 给随机 running 统计量），输入随机，64x64 图像；检测模型的 RoI 固定、RPN 头输出计入比对，proposal 选择 / NMS 因形状数据相关未导出。可变形卷积与 PS-RoI 池化是 `../dcn_ops.py` 中的纯张量实现（torch-mlir 无法 lower torchvision 的自定义算子），与 torchvision.ops 数值一致到 1e-7。

## 形状与规模

| | 值 |
|---|---|
| 输入 | 1x196x16x16（50,176 个 float） |
| 输出元素数 | 588 |
| 参数量 | 0.0M |
| 权重 blob | 0 MB |

## 文件

| 文件 | 内容 |
|---|---|
| `deform_psroi_tv.s` | hwacha-cc 生成的汇编，入口 `net` |
| `deform_psroi_tv_weights.bin.S` | 权重的 `.incbin` 桩（`split_weights.py` 分成 `.weights_lo` / `.weights_hi` 两段） |
| `deform_psroi_tv_weights.bin` | 权重 blob（不入 git，`make gen-deform_psroi` 按固定种子重建） |
| `deform_psroi_tv_check.bin` | 输入与 PyTorch 参考输出，host 用 `.incbin` 内嵌 |
| `tv_main.c` | 通用 host：调用 `net`，比对 max\|diff\|（分类型输出还比对 argmax） |
| `hwlib.s` | 卷积 / 池化库内核 |
| `HWMLIRFLAGS` | 本 case 需要的 hwacha-mlir 映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make deform_psroi          # 编译 -> deform_psroi/deform_psroi_tv.riscv
make deform_psroi.spike    # 在 Spike 上运行
make gen-deform_psroi      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`。

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 299,888 |
| max\|diff\| | 0 |
| max\|ref\| | 1.65546 |
| argmax（硬件 / 参考） | 387 / 387 |

