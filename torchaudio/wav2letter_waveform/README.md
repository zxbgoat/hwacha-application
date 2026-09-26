# wav2letter_waveform

torchaudio 的 `torchaudio.models.Wav2Letter` 在 Hwacha 上的一次前向，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/audio/stable/generated/torchaudio.models.Wav2Letter.html

用例：Wav2Letter(input_type='waveform')：多一层核 250、stride 160 的输入卷积，输入 1000 个采样，输出 (1, 10, 4)。

**注意**：架构不可缩小，IR 约 210 MB。参考值由真正的 torchaudio 前向算出，导出前脚本断言两者一致。

来源：`export_ta.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机权重与输入、固定种子；页面上的工厂函数构建论文尺寸的模型，远超 Spike 的规模，这里用微型尺寸实例化同一架构。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x1x1000 |
| 输出 | 1x10x4 |

## 文件

| 文件 | 内容 |
|---|---|
| `wav2letter_waveform.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `wav2letter_waveform_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make wav2letter_waveform          # 编译 -> wav2letter_waveform/wav2letter_waveform.riscv
make wav2letter_waveform.spike    # 在 Spike 上运行
make gen-wav2letter_waveform      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 3,404,042 |
| max\|diff\| | 0 |
| max\|ref\| | 2.30621 |

