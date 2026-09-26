# spectral_centroid

torchaudio 的 `torchaudio.transforms.SpectralCentroid` 在 Hwacha 上的一次应用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.SpectralCentroid.html

用例：SpectralCentroid(采样率 800, n_fft 32, hop 8)。

**注意**：torch.stft 经 view_as_real 可以导出，但复数的 abs / istft / angle / conj 没有 lowering；导出图：幅度谱 sqrt(re² + im²)，频率加权平均。参考值由真正的 torchaudio 变换算出，导出前脚本断言两者一致。

来源：`export_taf.py` / `taf_lib.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子；复数谱以末维为 (re, im) 的实张量进出。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x128 |
| 输出 | 1x17 |

## 文件

| 文件 | 内容 |
|---|---|
| `spectral_centroid.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `spectral_centroid_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make spectral_centroid          # 编译 -> spectral_centroid/spectral_centroid.riscv
make spectral_centroid.spike    # 在 Spike 上运行
make gen-spectral_centroid      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 3,512 |
| max\|diff\| | 4.5e-05 |
| max\|ref\| | 239.582 |

