# tacotron2

torchaudio 的 `torchaudio.models.Tacotron2` 在 Hwacha 上的一次前向，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/audio/stable/generated/torchaudio.models.Tacotron2.html

用例：Tacotron2(n_mels 8, 12 个符号, 嵌入 / 编码 16, 编码器 2 层核 3 卷积, 解码器 / 注意力 RNN 16, 位置注意力 4 滤波器核 5, prenet 8, postnet 2 层)，teacher forcing：输入 6 个 token，mel (1, 8, 5) 为常量，输出 [mel | mel_postnet | gate | alignments]。

**注意**：编码器用 pack_padded_sequence（无导出路径），解码器由 lengths 生成 memory 掩码（数据相关），prenet 在 eval 下也做 dropout(p=0.5, training=True)。导出图：编码器的卷积 + LSTM 直接对全长序列计算，memory 掩码固定为全 False，prenet 的 dropout 关闭（参考同样关闭 dropout：否则参考本身不可复现）。参考值由真正的 torchaudio 前向算出，导出前脚本断言两者一致。

来源：`export_ta.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机权重与输入、固定种子；页面上的工厂函数构建论文尺寸的模型，远超 Spike 的规模，这里用微型尺寸实例化同一架构。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x6 |
| 输出 | 115 |

## 文件

| 文件 | 内容 |
|---|---|
| `tacotron2.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `tacotron2_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make tacotron2          # 编译 -> tacotron2/tacotron2.riscv
make tacotron2.spike    # 在 Spike 上运行
make gen-tacotron2      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 47,271 |
| max\|diff\| | 0 |
| max\|ref\| | 0.627078 |

