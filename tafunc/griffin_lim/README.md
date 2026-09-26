# griffin_lim

torchaudio 的 `torchaudio.transforms.GriffinLim` 在 Hwacha 上的一次应用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.GriffinLim.html

用例：GriffinLim(n_fft 32, hop 8, n_iter 4, rand_init=False, length 128)：从功率谱重建波形。

**注意**：torch.stft 经 view_as_real 可以导出，但复数的 abs / istft / angle / conj 没有 lowering。导出图按 torchaudio.functional.griffinlim 展开 4 次迭代（动量 0.99/(1+0.99)，相位初始为 1），复数为 (re, im) 对，逆 STFT 是矩阵乘：帧 = Z @ 加 c_f 权重的逆 DFT 矩阵，乘窗，用 0/1 矩阵 overlap-add，除以窗平方和（torch.istft 的归一化），再裁掉 n_fft/2 与到 length。参考值由真正的 torchaudio 变换算出，导出前脚本断言两者一致。

来源：`export_taf.py` / `taf_lib.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子；复数谱以末维为 (re, im) 的实张量进出。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x17x17 |
| 输出 | 1x128 |

## 文件

| 文件 | 内容 |
|---|---|
| `griffin_lim.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `griffin_lim_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make griffin_lim          # 编译 -> griffin_lim/griffin_lim.riscv
make griffin_lim.spike    # 在 Spike 上运行
make gen-griffin_lim      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 31,837 |
| max\|diff\| | 2e-06 |
| max\|ref\| | 1.30721 |

