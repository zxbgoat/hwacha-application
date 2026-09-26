# souden_mvdr

torchaudio 的 `torchaudio.transforms.SoudenMVDR` 在 Hwacha 上的一次应用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.SoudenMVDR.html

用例：SoudenMVDR：常量 psd_s / psd_n -> 权重 -> 波束形成。

**注意**：同 mvdr 的实数 2x2 复矩阵算术。参考值由真正的 torchaudio 变换算出，导出前脚本断言两者一致。

来源：`export_taf.py` / `taf_lib.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子；复数谱以末维为 (re, im) 的实张量进出。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 2x17x17x2 |
| 输出 | 17x17x2 |

## 文件

| 文件 | 内容 |
|---|---|
| `souden_mvdr.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `souden_mvdr_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make souden_mvdr          # 编译 -> souden_mvdr/souden_mvdr.riscv
make souden_mvdr.spike    # 在 Spike 上运行
make gen-souden_mvdr      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 9,393 |
| max\|diff\| | 1e-06 |
| max\|ref\| | 8.78615 |

