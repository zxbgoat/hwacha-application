# vision_transformer

torchtune 的 `torchtune.modules.VisionTransformer` 在 Hwacha 上的一次前向，与 PyTorch 逐元素比对。文档：https://meta-pytorch.org/torchtune/0.6/generated/torchtune.modules.VisionTransformer.html

用例：1 张 8x8 图、patch 4 -> 4 个 patch + CLS，1 层 transformer，输出 token 序列 (1, 1, 1, 5, 32)。

来源：`export_ttf.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机权重与输入、固定种子；规模很小（embed 32、4 头 x head_dim 8、序列 8、词表 16）。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x1x1x3x8x8 |
| 输出 | 1x1x1x5x32 |

## 文件

| 文件 | 内容 |
|---|---|
| `vision_transformer.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `vision_transformer_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `hwlib.s` | 库内核 |
| `README.md` | 本文件 |

## 编译与运行

```
make vision_transformer          # 编译 -> vision_transformer/vision_transformer.riscv
make vision_transformer.spike    # 在 Spike 上运行
make gen-vision_transformer      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 12,414 |
| max\|diff\| | 0 |
| max\|ref\| | 2.67955 |

