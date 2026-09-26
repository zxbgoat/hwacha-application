# frequency_masking

torchaudio 的 `torchaudio.transforms.FrequencyMasking` 在 Hwacha 上的一次应用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.FrequencyMasking.html

用例：FrequencyMasking(freq_mask_param 6) 在 seed 0 下的一次抽样。

**注意**：掩码的起点和宽度在 forward 里随机抽取（数据相关）；掩码固定为 seed 0 下对全 1 张量应用得到的 0/1 模式，导出图乘以该掩码，参考在同一 seed 下调用。参考值由真正的 torchaudio 变换算出，导出前脚本断言两者一致。

来源：`export_taf.py` / `taf_lib.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子；复数谱以末维为 (re, im) 的实张量进出。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x17x17 |
| 输出 | 1x17x17 |

## 文件

| 文件 | 内容 |
|---|---|
| `frequency_masking.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `frequency_masking_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make frequency_masking          # 编译 -> frequency_masking/frequency_masking.riscv
make frequency_masking.spike    # 在 Spike 上运行
make gen-frequency_masking      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 78 |
| max\|diff\| | 0 |
| max\|ref\| | 18.9538 |

