# token_positional_embedding

torchtune 模型参考页的 `torchtune.models.clip.TokenPositionalEmbedding` 所用架构在 Hwacha 上的一次前向，与 PyTorch 逐元素比对。文档：https://meta-pytorch.org/torchtune/0.6/generated/torchtune.models.clip.TokenPositionalEmbedding.html

用例：单 tile 的 token 位置编码：x + 可学习的 (5, 32) 表。

来源：`export_ttm.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机权重与输入、固定种子。带尺寸的构建函数（7b、0.5b …）是同一组件构建函数加固定超参数，远超 Spike 的规模，这里用微型尺寸实例化同一架构。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x1x1x5x32 |
| 输出 | 1x1x1x5x32 |

## 文件

| 文件 | 内容 |
|---|---|
| `token_positional_embedding.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `token_positional_embedding_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make token_positional_embedding          # 编译 -> token_positional_embedding/token_positional_embedding.riscv
make token_positional_embedding.spike    # 在 Spike 上运行
make gen-token_positional_embedding      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 76 |
| max\|diff\| | 0 |
| max\|ref\| | 3.15845 |

