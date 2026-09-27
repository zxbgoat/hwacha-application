# cummin

torch 的 `torch.cummin`（Other Operations）在 Hwacha 上的一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/generated/torch.cummin.html

定义（`export_op.py`）：

```python
rcase('cummin', lambda s, x: -LB.cummax_rows(-x), lambda s, x: torch.cummin(x, 1).values, X)
```

输入为 4x8 随机张量（算子要求时为整数 / 布尔 / 正数 / 有界输入或别的形状）；第二操作数、下标、掩码、权重为常量 buffer，整数 / 布尔结果转为 float，多个结果拉平拼接，in-place 变体作用于输入副本并返回它，复数张量以末维 [re | im] 表示，随机抽样在 seed 0 下抽一次（参考在同一 seed 下抽取）。

**注意**：导出图与 torch 的算子不同（该算子没有 torch-mlir / hwacha-mlir 的 lowering，或 hwacha-cc 没有对应的向量 libm 函数，或输出形状数据相关）；等价的张量写法见 `op_lib.py` 与套件 README 的说明。参考值由真正的 torch 调用算出，导出前脚本断言两者一致。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 4x8 |
| 输出 | 4x8 |

## 文件

| 文件 | 内容 |
|---|---|
| `cummin.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `cummin_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make cummin          # 编译 -> cummin/cummin.riscv
make cummin.spike    # 在 Spike 上运行
make gen-cummin      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 684 |
| max\|diff\| | 0 |
| max\|ref\| | 2.11522 |

