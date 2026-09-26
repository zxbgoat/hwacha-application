# rnnt

torchaudio 的 `torchaudio.models.RNNT` 在 Hwacha 上的一次前向，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/audio/stable/generated/torchaudio.models.RNNT.html

用例：emformer_rnnt_model(...)：Emformer 转写器（时间缩减 2、segment 4）+ LSTM 预测器（符号嵌入 8、1 层、LayerNorm）+ 联合网络（10 个符号），输入 20 帧 x 16 与 4 个目标符号，输出联合网络的 (1, 9, 5, 10)。

**注意**：同 emformer：分解表去掉张量构造算子。参考值由真正的 torchaudio 前向算出，导出前脚本断言两者一致。

来源：`export_ta.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机权重与输入、固定种子；页面上的工厂函数构建论文尺寸的模型，远超 Spike 的规模，这里用微型尺寸实例化同一架构。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x20x16 |
| 输出 | 1x9x4x10 |

## 文件

| 文件 | 内容 |
|---|---|
| `rnnt.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `rnnt_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make rnnt          # 编译 -> rnnt/rnnt.riscv
make rnnt.spike    # 在 Spike 上运行
make gen-rnnt      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 37,310 |
| max\|diff\| | 0 |
| max\|ref\| | 1.18583 |

