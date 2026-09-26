# rnnt_loss

torchaudio 的 `torchaudio.transforms.RNNTLoss` 在 Hwacha 上的一次应用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.RNNTLoss.html

用例：RNNTLoss(blank 0)：(1, 4, 3, 5) 的 logits、目标 [1, 2]。

**注意**：C++ 损失没有导出路径；导出图把 transducer 的 alpha 递推在 4 x 3 格点上展开（fused log-softmax，logaddexp）。参考值由真正的 torchaudio 变换算出，导出前脚本断言两者一致。

来源：`export_taf.py` / `taf_lib.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子；复数谱以末维为 (re, im) 的实张量进出。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x4x3x5 |
| 输出 | 1 |

## 文件

| 文件 | 内容 |
|---|---|
| `rnnt_loss.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `rnnt_loss_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make rnnt_loss          # 编译 -> rnnt_loss/rnnt_loss.riscv
make rnnt_loss.spike    # 在 Spike 上运行
make gen-rnnt_loss      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 1,949 |
| max\|diff\| | 0 |
| max\|ref\| | 9.4744 |

