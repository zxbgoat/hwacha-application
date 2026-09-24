# cosinesimilarity

`torch.nn` 单层测试：**TwoIn(nn.CosineSimilarity(dim=1),(4,8))**，一次前向，与 PyTorch 逐元素比对。

输入：`torch.randn(4,8)`。

包装说明：`TwoIn(m, shape)`：第二个输入（memory / tgt）固定为随机常量 buffer。

来源：`hwacha-cc/test/modules/export_module.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机权重、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 4x8 |
| 输出 | 4 |
| 常量 buffer | `second` 4x8 |

## 文件

| 文件 | 内容 |
|---|---|
| `cosinesimilarity.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `cosinesimilarity_check.bin` | 输入与 PyTorch 参考输出（int32 个数 + float32 数组，各一段），host 用 `.incbin` 内嵌 |
| `README.md` | 本文件 |

## 编译与运行

```
make cosinesimilarity          # 编译 -> cosinesimilarity/cosinesimilarity.riscv
make cosinesimilarity.spike    # 在 Spike 上运行
```

hwacha-mlir 映射：`--collapse-all`（所有并行维映射到 lane）。

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 799 |
| max\|diff\| | 0 |
| max\|ref\| | 0.642947 |

