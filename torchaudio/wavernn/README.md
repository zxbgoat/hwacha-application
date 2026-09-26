# wavernn

torchaudio 的 `torchaudio.models.WaveRNN` 在 Hwacha 上的一次前向，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/audio/stable/generated/torchaudio.models.WaveRNN.html

用例：WaveRNN(上采样 [2, 2], 16 类, hop 4, 1 个残差块, RNN / FC 16, 核 3, 8 频带, hidden / output 8)：输入 16 个采样与 (1, 1, 8, 6) 的频谱常量，输出每个采样 16 类的 logits (1, 1, 16, 16)。

来源：`export_ta.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机权重与输入、固定种子；页面上的工厂函数构建论文尺寸的模型，远超 Spike 的规模，这里用微型尺寸实例化同一架构。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x1x16 |
| 输出 | 1x1x16x16 |

## 文件

| 文件 | 内容 |
|---|---|
| `wavernn.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `wavernn_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make wavernn          # 编译 -> wavernn/wavernn.riscv
make wavernn.spike    # 在 Spike 上运行
make gen-wavernn      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 36,283 |
| max\|diff\| | 0 |
| max\|ref\| | 0.518245 |

