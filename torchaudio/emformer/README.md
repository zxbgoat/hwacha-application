# emformer

torchaudio 的 `torchaudio.models.Emformer` 在 Hwacha 上的一次前向，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/audio/stable/generated/torchaudio.models.Emformer.html

用例：Emformer(16, 2 头, ffn 32, 2 层, segment 4, 右上下文 2, 左上下文 4, 记忆 1)，输入 12 + 2 帧，输出 (1, 12, 16)。

**注意**：torch-mlir 默认的分解表把 aten.zeros / ones（Emformer 的注意力掩码）分解成没有 lowering 的 aten.empty_strided；导出时从分解表里去掉张量构造算子（torch-mlir 直接 lower aten.zeros / ones / full）。参考值由真正的 torchaudio 前向算出，导出前脚本断言两者一致。

来源：`export_ta.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机权重与输入、固定种子；页面上的工厂函数构建论文尺寸的模型，远超 Spike 的规模，这里用微型尺寸实例化同一架构。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x14x16 |
| 输出 | 1x12x16 |

## 文件

| 文件 | 内容 |
|---|---|
| `emformer.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `emformer_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make emformer          # 编译 -> emformer/emformer.riscv
make emformer.spike    # 在 Spike 上运行
make gen-emformer      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 21,587 |
| max\|diff\| | 0 |
| max\|ref\| | 2.76313 |

