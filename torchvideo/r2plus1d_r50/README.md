# r2plus1d_r50

PyTorchVideo model zoo（https://pytorchvideo.readthedocs.io/en/latest/model_zoo.html）的 **R(2+1)D R50**（Kinetics-400, 16x4）在 Hwacha 上的一次前向，与 PyTorch 比对。构建：`pytorchvideo.models.hub.r2plus1d_r50`。

模型：每个 3x3x3 卷积分解为 1x3x3 空间卷积 + 3x1x1 时间卷积（中间通道数按论文公式）。

比对内容：400 类 logits，逐元素比对（容差 1e-4 + 1e-2·max\|ref\|），另比对 argmax。

权重随机（固定种子，未下载 checkpoint；BatchNorm 给随机 running 统计量），输入随机。 输入 1x3x16x32x32（帧数为该设置的 frame length，空间 32x32 而非 224）。

**与 PyTorchVideo 的差别**：头部为 224x224 定尺寸的 AvgPool3d（在 /32 后的 7x7 图上即全局平均）换成 AdaptiveAvgPool3d(1)，其余不变。

## 形状与规模

| | 值 |
|---|---|
| 输入 | 1x3x16x32x32（49,152 个 float） |
| 输出元素数 | 400 |
| 参数量 | 28.11M |
| 权重 blob | 107 MB |

## 文件

| 文件 | 内容 |
|---|---|
| `r2plus1d_r50_tv.s` | hwacha-cc 生成的汇编，入口 `net` |
| `r2plus1d_r50_tv_weights.bin.S` | 权重的 `.incbin` 桩（`split_weights.py` 分成 `.weights_lo` / `.weights_hi` 两段） |
| `r2plus1d_r50_tv_weights.bin` | 权重 blob（不入 git，`make gen-r2plus1d_r50` 按固定种子重建） |
| `r2plus1d_r50_tv_check.bin` | 输入与 PyTorch 参考输出，host 用 `.incbin` 内嵌 |
| `tv_main.c` | 通用 host：调用 `net`，比对 max\|diff\|（分类型输出还比对 argmax） |
| `hwlib.s` | 卷积 / 池化库内核 |
| `HWMLIRFLAGS` | 本 case 需要的 hwacha-mlir 映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make r2plus1d_r50          # 编译 -> r2plus1d_r50/r2plus1d_r50_tv.riscv
make r2plus1d_r50.spike    # 在 Spike 上运行
make gen-r2plus1d_r50      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--unroll-small=2`。

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 3,162,653,849 |
| max\|diff\| | 1e-06 |
| max\|ref\| | 0.976116 |
| argmax（硬件 / 参考） | 36 / 36 |

