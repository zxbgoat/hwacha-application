# mvit_base_32x3

PyTorchVideo model zoo（https://pytorchvideo.readthedocs.io/en/latest/model_zoo.html）的 **MViT B**（Kinetics-400, 32x3）在 Hwacha 上的一次前向，与 PyTorch 比对。构建：`pytorchvideo.models.hub.mvit_base_32x3`。

模型：同 mvit_base_16x4，32 帧输入。

比对内容：400 类 logits，逐元素比对（容差 1e-4 + 1e-2·max\|ref\|），另比对 argmax。

权重随机（固定种子，未下载 checkpoint；BatchNorm 给随机 running 统计量），输入随机。 输入 1x3x32x32x32（帧数为该设置的 frame length，空间 32x32 而非 224）。

**与 PyTorchVideo 的差别**：cls token 与位置编码在 forward 里由 nn.Parameter 计算再 torch.cat，torch-mlir 的导入器拒绝（aten.cat 列表里的 Parameter）；eval 下两者都是常量，预先算成 buffer。

## 形状与规模

| | 值 |
|---|---|
| 输入 | 1x3x32x32x32（98,304 个 float） |
| 输出元素数 | 400 |
| 参数量 | 36.32M |
| 权重 blob | 139 MB |

## 文件

| 文件 | 内容 |
|---|---|
| `mvit_base_32x3_tv.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mvit_base_32x3_tv_weights.bin.S` | 权重的 `.incbin` 桩（`split_weights.py` 分成 `.weights_lo` / `.weights_hi` 两段） |
| `mvit_base_32x3_tv_weights.bin` | 权重 blob（不入 git，`make gen-mvit_base_32x3` 按固定种子重建） |
| `mvit_base_32x3_tv_check.bin` | 输入与 PyTorch 参考输出，host 用 `.incbin` 内嵌 |
| `tv_main.c` | 通用 host：调用 `net`，比对 max\|diff\|（分类型输出还比对 argmax） |
| `hwlib.s` | 卷积 / 池化库内核 |
| `HWMLIRFLAGS` | 本 case 需要的 hwacha-mlir 映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make mvit_base_32x3          # 编译 -> mvit_base_32x3/mvit_base_32x3_tv.riscv
make mvit_base_32x3.spike    # 在 Spike 上运行
make gen-mvit_base_32x3      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--unroll-small=2`。

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 187,781,550 |
| max\|diff\| | 2e-06 |
| max\|ref\| | 1.76965 |
| argmax（硬件 / 参考） | 241 / 241 |

