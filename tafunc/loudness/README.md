# loudness

torchaudio 的 `torchaudio.transforms.Loudness` 在 Hwacha 上的一次应用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.Loudness.html

用例：Loudness(采样率 200)：ITU-R BS.1770 响度（LKFS），400 个采样。

**注意**：K 加权的两个 biquad 是 lfilter（导出下形状出错）；导出图用两个冲激响应矩阵乘（各自截到 [-1, 1]，与 lfilter 一致），400 ms 块用常量分帧下标，两级门控用掩码均值；阈值 / 对数与 torchaudio.functional.loudness 相同。参考值由真正的 torchaudio 变换算出，导出前脚本断言两者一致。

来源：`export_taf.py` / `taf_lib.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子；复数谱以末维为 (re, im) 的实张量进出。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x400 |
| 输出 | 1 |

## 文件

| 文件 | 内容 |
|---|---|
| `loudness.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `loudness_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make loudness          # 编译 -> loudness/loudness.riscv
make loudness.spike    # 在 Spike 上运行
make gen-loudness      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 8,127 |
| max\|diff\| | 6e-05 |
| max\|ref\| | 13.3666 |

