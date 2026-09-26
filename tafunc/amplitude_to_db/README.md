# amplitude_to_db

torchaudio 的 `torchaudio.transforms.AmplitudeToDB` 在 Hwacha 上的一次应用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.AmplitudeToDB.html

用例：AmplitudeToDB('power', top_db 80) 作用于功率谱。

来源：`export_taf.py` / `taf_lib.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子；复数谱以末维为 (re, im) 的实张量进出。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x17x17 |
| 输出 | 1x17x17 |

## 文件

| 文件 | 内容 |
|---|---|
| `amplitude_to_db.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `amplitude_to_db_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make amplitude_to_db          # 编译 -> amplitude_to_db/amplitude_to_db.riscv
make amplitude_to_db.spike    # 在 Spike 上运行
make gen-amplitude_to_db      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 1,033 |
| max\|diff\| | 1e-06 |
| max\|ref\| | 43.3199 |

