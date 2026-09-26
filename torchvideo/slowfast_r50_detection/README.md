# slowfast_r50_detection

PyTorchVideo model zoo（https://pytorchvideo.readthedocs.io/en/latest/model_zoo.html）的 **SlowFast R50 detection**（AVA v2.2, 8x8）在 Hwacha 上的一次前向，与 PyTorch 比对。构建：`pytorchvideo.models.hub.slowfast_r50_detection`。

模型：SlowFast R50 主干（stride 16）+ 两路各自时间平均后拼接 -> RoIAlign 7x7 -> 最大池化 -> 线性 80 类 -> Sigmoid；2 个固定框。

比对内容：2 个框各 80 类的 sigmoid 得分，逐元素比对（容差 1e-4 + 1e-2·max\|ref\|）。

权重随机（固定种子，未下载 checkpoint；BatchNorm 给随机 running 统计量），输入随机。 输入 1x3x40x64x64：Slow 与 Fast 两段 clip 沿时间轴拼成一个张量（前 T 帧 Slow、后 4T 帧 Fast），网络入口拆开。

**与 PyTorchVideo 的差别**：头部为 224x224 定尺寸的 AvgPool3d（在 /32 后的 7x7 图上即全局平均）换成 AdaptiveAvgPool3d(1)，其余不变；RoIAlign 无 torch-mlir lowering；框固定，按 torchvision CUDA 内核的采样规则（sampling_ratio=0 -> 每 bin ceil(roi/7)² 个双线性采样点，越界一像素内夹到边界）预先算成 (R·49, H·W) 的常量 gather 矩阵，RoI 池化即一次矩阵乘，与 torchvision.ops.RoIAlign 一致到 1e-5。

## 形状与规模

| | 值 |
|---|---|
| 输入 | 1x3x40x64x64（491,520 个 float） |
| 输出元素数 | 160 |
| 参数量 | 33.83M |
| 权重 blob | 129 MB |

## 文件

| 文件 | 内容 |
|---|---|
| `slowfast_r50_detection_tv.s` | hwacha-cc 生成的汇编，入口 `net` |
| `slowfast_r50_detection_tv_weights.bin.S` | 权重的 `.incbin` 桩（`split_weights.py` 分成 `.weights_lo` / `.weights_hi` 两段） |
| `slowfast_r50_detection_tv_weights.bin` | 权重 blob（不入 git，`make gen-slowfast_r50_detection` 按固定种子重建） |
| `slowfast_r50_detection_tv_check.bin` | 输入与 PyTorch 参考输出，host 用 `.incbin` 内嵌 |
| `tv_main.c` | 通用 host：调用 `net`，比对 max\|diff\|（分类型输出还比对 argmax） |
| `hwlib.s` | 卷积 / 池化库内核 |
| `HWMLIRFLAGS` | 本 case 需要的 hwacha-mlir 映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make slowfast_r50_detection          # 编译 -> slowfast_r50_detection/slowfast_r50_detection_tv.riscv
make slowfast_r50_detection.spike    # 在 Spike 上运行
make gen-slowfast_r50_detection      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--unroll-small=2`。

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 14,601,039,989 |
| max\|diff\| | 1.3e-05 |
| max\|ref\| | 1 |
| argmax（硬件 / 参考） | 0 / 0 |

