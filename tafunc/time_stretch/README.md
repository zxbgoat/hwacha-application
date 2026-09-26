# time_stretch

torchaudio 的 `torchaudio.transforms.TimeStretch` 在 Hwacha 上的一次应用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.TimeStretch.html

用例：TimeStretch(hop 8, n_freq 17, rate 1.25)：相位声码器，复数谱 (1, 17, 17, 2) -> (1, 17, 14, 2)。

**注意**：aten.angle 与复数运算没有 lowering；导出图按 torchaudio.functional.phase_vocoder 写实数版：角度用 atan 加象限修正，帧插值用常量 index_select，相位累积 cumsum 用上三角矩阵乘，polar 用 cos / sin。参考值由真正的 torchaudio 变换算出，导出前脚本断言两者一致。

来源：`export_taf.py` / `taf_lib.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子；复数谱以末维为 (re, im) 的实张量进出。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x17x17x2 |
| 输出 | 1x17x14x2 |

## 文件

| 文件 | 内容 |
|---|---|
| `time_stretch.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `time_stretch_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make time_stretch          # 编译 -> time_stretch/time_stretch.riscv
make time_stretch.spike    # 在 Spike 上运行
make gen-time_stretch      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 3,887 |
| max\|diff\| | 0.000102 |
| max\|ref\| | 4.01831 |

