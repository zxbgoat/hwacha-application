# hdemucs

torchaudio 的 `torchaudio.models.HDemucs` 在 Hwacha 上的一次前向，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/audio/stable/generated/torchaudio.models.HDemucs.html

用例：HDemucs(2 源, 单声道, channels 4, nfft 64, depth 2)：混合时频域源分离，输入 256 个采样，输出 (1, 2, 1, 256)。

**注意**：torch.stft / istft、hann_window 与复数张量都没有 lowering。导出用 HDemucs 的子类替换 _spec / _magnitude / _mask / _ispec：频谱是末维为 (re, im) 的实张量，帧用常量下标 gather、乘加窗的 DFT 矩阵（normalized、center、reflect 填充与 torch.stft 一致），逆变换是加窗的逆 DFT 矩阵乘、overlap-add 矩阵乘和 torch.istft 的窗平方和归一化；权重与参考模型相同（deepcopy），容差 2e-3。参考值由真正的 torchaudio 前向算出，导出前脚本断言两者一致。

来源：`export_ta.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机权重与输入、固定种子；页面上的工厂函数构建论文尺寸的模型，远超 Spike 的规模，这里用微型尺寸实例化同一架构。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x1x256 |
| 输出 | 1x2x1x256 |

## 文件

| 文件 | 内容 |
|---|---|
| `hdemucs.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `hdemucs_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `hwlib.s` | 库内核 |
| `README.md` | 本文件 |

## 编译与运行

```
make hdemucs          # 编译 -> hdemucs/hdemucs.riscv
make hdemucs.spike    # 在 Spike 上运行
make gen-hdemucs      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 118,908 |
| max\|diff\| | 0 |
| max\|ref\| | 0.271036 |

