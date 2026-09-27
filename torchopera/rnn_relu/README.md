# rnn_relu

torch 的 `torch.rnn_relu`（Other Operations）在 Hwacha 上的一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/generated/torch.rnn_relu.html

定义（`export_op.py`）：

```python
case('rnn_relu', lambda s, x: flat(*torch.rnn_relu(x, s.h, [s.wih, s.whh, s.bih, s.bhh], True, 1, 0.0, False, False, False)), R(5, 2, 4), h=R(1, 2, 3), wih=R(3, 4), whh=R(3, 3), bih=R(3), bhh=R(3))
```

输入为 4x8 随机张量（算子要求时为整数 / 布尔 / 正数 / 有界输入或别的形状）；第二操作数、下标、掩码、权重为常量 buffer，整数 / 布尔结果转为 float，多个结果拉平拼接，in-place 变体作用于输入副本并返回它，复数张量以末维 [re | im] 表示，随机抽样在 seed 0 下抽一次（参考在同一 seed 下抽取）。

常量 buffer：`h` 1x2x3, `wih` 3x4, `whh` 3x3, `bih` 3, `bhh` 3

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 5x2x4 |
| 输出 | 36 |

## 文件

| 文件 | 内容 |
|---|---|
| `rnn_relu.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `rnn_relu_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make rnn_relu          # 编译 -> rnn_relu/rnn_relu.riscv
make rnn_relu.spike    # 在 Spike 上运行
make gen-rnn_relu      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 2,023 |
| max\|diff\| | 0 |
| max\|ref\| | 4.45725 |

