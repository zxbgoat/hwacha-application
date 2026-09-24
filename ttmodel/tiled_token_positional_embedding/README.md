# tiled_token_positional_embedding

torchtune 模型参考页的 `torchtune.models.clip.TiledTokenPositionalEmbedding` 所用架构在 Hwacha 上的一次前向，与 PyTorch 逐元素比对。文档：https://meta-pytorch.org/torchtune/0.6/generated/torchtune.models.clip.TiledTokenPositionalEmbedding.html

用例：平铺 token 位置编码：2 个 tile（aspect ratio (2, 1)），局部表 + 门控的全局表。

**注意**：前向按每张图的 aspect ratio 循环并原地累加（数据相关）；固定 aspect ratio (2, 1) 下导出图加上同样的常量切片 local·(1 - tanh g) + global[:2, :1]·tanh g。参考值由真正的 torchtune 前向算出，导出前脚本断言两者一致。

来源：`export_ttm.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机权重与输入、固定种子。带尺寸的构建函数（7b、0.5b …）是同一组件构建函数加固定超参数，远超 Spike 的规模，这里用微型尺寸实例化同一架构。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x2x5x32 |
| 输出 | 1x2x5x32 |

## 文件

| 文件 | 内容 |
|---|---|
| `tiled_token_positional_embedding.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `tiled_token_positional_embedding_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make tiled_token_positional_embedding          # 编译 -> tiled_token_positional_embedding/tiled_token_positional_embedding.riscv
make tiled_token_positional_embedding.spike    # 在 Spike 上运行
make gen-tiled_token_positional_embedding      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 357 |
| max\|diff\| | 0 |
| max\|ref\| | 2.98007 |

