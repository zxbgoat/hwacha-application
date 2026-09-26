# spectrogram_complex

torchaudio 的 `torchaudio.transforms.Spectrogram` 在 Hwacha 上的一次应用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.Spectrogram.html

用例：Spectrogram(power=None)：复数谱，输出 view_as_real 的 (1, 17, 17, 2)。

来源：`export_taf.py` / `taf_lib.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子；复数谱以末维为 (re, im) 的实张量进出。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x128 |
| 输出 | 1x17x17x2 |

## 文件

| 文件 | 内容 |
|---|---|
| `spectrogram_complex.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `spectrogram_complex_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make spectrogram_complex          # 编译 -> spectrogram_complex/spectrogram_complex.riscv
make spectrogram_complex.spike    # 在 Spike 上运行
make gen-spectrogram_complex      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 1,444 |
| max\|diff\| | 0 |
| max\|ref\| | 4.35359 |

