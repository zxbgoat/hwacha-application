# x3d_xs

PyTorchVideo model zoo（https://pytorchvideo.readthedocs.io/en/latest/model_zoo.html）的 **X3D XS**（Kinetics-400, 4x12）在 Hwacha 上的一次前向，与 PyTorch 比对。构建：`pytorchvideo.models.x3d.create_x3d(input_clip_length=4)`。

模型：X3D：逐通道 3x3x3 卷积 + SE + Swish 的 bottleneck，宽度因子 2.0、深度因子 2.2；head 为 1x1 卷积 -> 池化 -> 1x1 卷积 -> 线性层。

比对内容：400 类 logits，逐元素比对（容差 1e-4 + 1e-2·max\|ref\|），另比对 argmax。

权重随机（固定种子，未下载 checkpoint；BatchNorm 给随机 running 统计量），输入随机。 输入 1x3x4x32x32（帧数为该设置的 frame length，空间 32x32 而非 224）。

**与 PyTorchVideo 的差别**：`create_x3d` 由 (input_clip_length, input_crop_size) 推出头部池化核，这里按 32x32 构建（hub 的 x3d_* 固定 crop 160 / 224 / 312）；逐通道 3-D 卷积（groups = 通道数）torch-mlir 标为 illegal（5-D 输入的分组 aten.convolution）；导出图把它按时间核的每个 tap 拆成对 B·T 帧的 2-D 逐通道卷积再按时间偏移求和（`DepthwiseConv3d`），与 nn.Conv3d 一致到 1e-7。

## 形状与规模

| | 值 |
|---|---|
| 输入 | 1x3x4x32x32（12,288 个 float） |
| 输出元素数 | 400 |
| 参数量 | 3.79M |
| 权重 blob | 15 MB |

## 文件

| 文件 | 内容 |
|---|---|
| `x3d_xs_tv.s` | hwacha-cc 生成的汇编，入口 `net` |
| `x3d_xs_tv_weights.bin.S` | 权重的 `.incbin` 桩（`split_weights.py` 分成 `.weights_lo` / `.weights_hi` 两段） |
| `x3d_xs_tv_weights.bin` | 权重 blob（不入 git，`make gen-x3d_xs` 按固定种子重建） |
| `x3d_xs_tv_check.bin` | 输入与 PyTorch 参考输出，host 用 `.incbin` 内嵌 |
| `tv_main.c` | 通用 host：调用 `net`，比对 max\|diff\|（分类型输出还比对 argmax） |
| `hwlib.s` | 卷积 / 池化库内核 |
| `HWMLIRFLAGS` | 本 case 需要的 hwacha-mlir 映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make x3d_xs          # 编译 -> x3d_xs/x3d_xs_tv.riscv
make x3d_xs.spike    # 在 Spike 上运行
make gen-x3d_xs      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`。

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 44,777,884 |
| max\|diff\| | 0 |
| max\|ref\| | 0.003297 |
| argmax（硬件 / 参考） | 31 / 31 |

