# squim_objective

torchaudio 的 `torchaudio.models.SquimObjective` 在 Hwacha 上的一次前向，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/audio/stable/generated/torchaudio.models.SquimObjective.html

用例：squim_objective_model(feat 16, win 8, d_model 16, 2 头, hidden 16, 1 块, LSTM, chunk 5)：输入 256 个采样，输出 [STOI, PESQ, SI-SDR]。

来源：`export_ta.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机权重与输入、固定种子；页面上的工厂函数构建论文尺寸的模型，远超 Spike 的规模，这里用微型尺寸实例化同一架构。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x256 |
| 输出 | 3 |

## 文件

| 文件 | 内容 |
|---|---|
| `squim_objective.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `squim_objective_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `hwlib.s` | 库内核 |
| `README.md` | 本文件 |

## 编译与运行

```
make squim_objective          # 编译 -> squim_objective/squim_objective.riscv
make squim_objective.spike    # 在 Spike 上运行
make gen-squim_objective      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 306,128 |
| max\|diff\| | 0 |
| max\|ref\| | 3.06654 |

