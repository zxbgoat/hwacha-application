# chain_matmul

torch 的 `torch.chain_matmul`（BLAS and LAPACK Operations）在 Hwacha 上的一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/generated/torch.chain_matmul.html

定义（`export_op.py`）：

```python
case('chain_matmul', lambda s, x: torch.chain_matmul(x, s.a, s.b), X, a=R(8, 3), b=R(3, 2))
```

输入为 4x8 随机张量（算子要求时为整数 / 布尔 / 正数 / 有界输入或别的形状）；第二操作数、下标、掩码、权重为常量 buffer，整数 / 布尔结果转为 float，多个结果拉平拼接，in-place 变体作用于输入副本并返回它，复数张量以末维 [re | im] 表示，随机抽样在 seed 0 下抽一次（参考在同一 seed 下抽取）。

常量 buffer：`a` 8x3, `b` 3x2

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 4x8 |
| 输出 | 4x2 |

## 文件

| 文件 | 内容 |
|---|---|
| `chain_matmul.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `chain_matmul_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make chain_matmul          # 编译 -> chain_matmul/chain_matmul.riscv
make chain_matmul.spike    # 在 Spike 上运行
make gen-chain_matmul      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 268 |
| max\|diff\| | 0 |
| max\|ref\| | 6.6138 |

