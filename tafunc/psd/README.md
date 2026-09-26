# psd

torchaudio 的 `torchaudio.transforms.PSD` 在 Hwacha 上的一次应用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.PSD.html

用例：PSD()：2 通道复数谱 (2, 17, 17, 2) 与 (17, 17) 的时频掩码 -> (17, 2, 2, 2) 的功率谱密度矩阵。

**注意**：aten._conj 没有 lowering；导出图用实部 / 虚部写外积求和（normalize=True 的掩码归一）。参考值由真正的 torchaudio 变换算出，导出前脚本断言两者一致。

来源：`export_taf.py` / `taf_lib.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子；复数谱以末维为 (re, im) 的实张量进出。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 2x17x17x2 |
| 输出 | 17x2x2x2 |

## 文件

| 文件 | 内容 |
|---|---|
| `psd.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `psd_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make psd          # 编译 -> psd/psd.riscv
make psd.spike    # 在 Spike 上运行
make gen-psd      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 2,181 |
| max\|diff\| | 5e-06 |
| max\|ref\| | 29.7383 |

