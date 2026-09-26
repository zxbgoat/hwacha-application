# deemphasis

torchaudio 的 `torchaudio.transforms.Deemphasis` 在 Hwacha 上的一次应用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.Deemphasis.html

用例：Deemphasis(0.97)：IIR y[n] = x[n] + 0.97 y[n-1]。

**注意**：lfilter 的 Python 实现在 torch.export 下形状出错；导出图用冲激响应矩阵 y = x @ M（M[i, j] = h[j - i]，128 采样内精确），再按 lfilter 的约定截到 [-1, 1]。参考值由真正的 torchaudio 变换算出，导出前脚本断言两者一致。

来源：`export_taf.py` / `taf_lib.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子；复数谱以末维为 (re, im) 的实张量进出。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x128 |
| 输出 | 1x128 |

## 文件

| 文件 | 内容 |
|---|---|
| `deemphasis.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `deemphasis_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make deemphasis          # 编译 -> deemphasis/deemphasis.riscv
make deemphasis.spike    # 在 Spike 上运行
make gen-deemphasis      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 1,070 |
| max\|diff\| | 2e-06 |
| max\|ref\| | 1 |

