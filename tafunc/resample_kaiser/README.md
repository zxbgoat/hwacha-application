# resample_kaiser

torchaudio 的 `torchaudio.transforms.Resample` 在 Hwacha 上的一次应用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.Resample.html

用例：Resample(4 -> 5, sinc_interp_kaiser)。

来源：`export_taf.py` / `taf_lib.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子；复数谱以末维为 (re, im) 的实张量进出。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x128 |
| 输出 | 1x160 |

## 文件

| 文件 | 内容 |
|---|---|
| `resample_kaiser.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `resample_kaiser_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make resample_kaiser          # 编译 -> resample_kaiser/resample_kaiser.riscv
make resample_kaiser.spike    # 在 Spike 上运行
make gen-resample_kaiser      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 461 |
| max\|diff\| | 0 |
| max\|ref\| | 1.38348 |

