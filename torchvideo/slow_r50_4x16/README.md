# slow_r50_4x16

PyTorchVideo model zoo（https://pytorchvideo.readthedocs.io/en/latest/model_zoo.html）的 **Slow R50**（Kinetics-400, 4x16）在 Hwacha 上的一次前向，与 PyTorch 比对。构建：`pytorchvideo.models.hub.slow_r50`。

模型：同 slow_r50，4 帧输入（4x16 设置）。

比对内容：400 类 logits，逐元素比对（容差 1e-4 + 1e-2·max\|ref\|），另比对 argmax。

权重随机（固定种子，未下载 checkpoint；BatchNorm 给随机 running 统计量），输入随机。 输入 1x3x4x32x32（帧数为该设置的 frame length，空间 32x32 而非 224）。

**与 PyTorchVideo 的差别**：头部为 224x224 定尺寸的 AvgPool3d（在 /32 后的 7x7 图上即全局平均）换成 AdaptiveAvgPool3d(1)，其余不变。

## 形状与规模

| | 值 |
|---|---|
| 输入 | 1x3x4x32x32（12,288 个 float） |
| 输出元素数 | 400 |
| 参数量 | 32.45M |
| 权重 blob | 124 MB |

## 文件

| 文件 | 内容 |
|---|---|
| `slow_r50_4x16_tv.s` | hwacha-cc 生成的汇编，入口 `net` |
| `slow_r50_4x16_tv_weights.bin.S` | 权重的 `.incbin` 桩（`split_weights.py` 分成 `.weights_lo` / `.weights_hi` 两段） |
| `slow_r50_4x16_tv_weights.bin` | 权重 blob（不入 git，`make gen-slow_r50_4x16` 按固定种子重建） |
| `slow_r50_4x16_tv_check.bin` | 输入与 PyTorch 参考输出，host 用 `.incbin` 内嵌 |
| `tv_main.c` | 通用 host：调用 `net`，比对 max\|diff\|（分类型输出还比对 argmax） |
| `hwlib.s` | 卷积 / 池化库内核 |
| `HWMLIRFLAGS` | 本 case 需要的 hwacha-mlir 映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make slow_r50_4x16          # 编译 -> slow_r50_4x16/slow_r50_4x16_tv.riscv
make slow_r50_4x16.spike    # 在 Spike 上运行
make gen-slow_r50_4x16      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--unroll-small=2`。

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 1,836,314,385 |
| max\|diff\| | 5e-05 |
| max\|ref\| | 27.4814 |
| argmax（硬件 / 参考） | 181 / 181 |

