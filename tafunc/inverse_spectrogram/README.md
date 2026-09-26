# inverse_spectrogram

torchaudio 的 `torchaudio.transforms.InverseSpectrogram` 在 Hwacha 上的一次应用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.InverseSpectrogram.html

用例：InverseSpectrogram(n_fft 32, hop 8)：输入复数谱 (1, 17, 17, 2)，输出 128 个采样。

**注意**：aten.istft 没有 lowering；逆 STFT 是矩阵乘：帧 = Z @ 加 c_f 权重的逆 DFT 矩阵，乘窗，用 0/1 矩阵 overlap-add，除以窗平方和（torch.istft 的归一化），再裁掉 n_fft/2 与到 length。参考值由真正的 torchaudio 变换算出，导出前脚本断言两者一致。

来源：`export_taf.py` / `taf_lib.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子；复数谱以末维为 (re, im) 的实张量进出。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x17x17x2 |
| 输出 | 1x128 |

## 文件

| 文件 | 内容 |
|---|---|
| `inverse_spectrogram.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `inverse_spectrogram_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make inverse_spectrogram          # 编译 -> inverse_spectrogram/inverse_spectrogram.riscv
make inverse_spectrogram.spike    # 在 Spike 上运行
make gen-inverse_spectrogram      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 4,947 |
| max\|diff\| | 0 |
| max\|ref\| | 1.70525 |

