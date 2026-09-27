# native_batch_norm

torch 的 `torch.native_batch_norm`（Other Operations）在 Hwacha 上的一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/generated/torch.native_batch_norm.html

定义（`export_op.py`）：

```python
case('native_batch_norm', lambda s, x: flat(*torch.native_batch_norm(x, s.w, s.b, s.rm, s.rv, False, 0.1, 1e-5)), R(2, 4, 8), w=R(4), b=R(4), rm=R(4), rv=U(4) + 0.5)
```

输入为 4x8 随机张量（算子要求时为整数 / 布尔 / 正数 / 有界输入或别的形状）；第二操作数、下标、掩码、权重为常量 buffer，整数 / 布尔结果转为 float，多个结果拉平拼接，in-place 变体作用于输入副本并返回它，复数张量以末维 [re | im] 表示，随机抽样在 seed 0 下抽一次（参考在同一 seed 下抽取）。

常量 buffer：`w` 4, `b` 4, `rm` 4, `rv` 4

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 2x4x8 |
| 输出 | 64 |

## 文件

| 文件 | 内容 |
|---|---|
| `native_batch_norm.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `native_batch_norm_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make native_batch_norm          # 编译 -> native_batch_norm/native_batch_norm.riscv
make native_batch_norm.spike    # 在 Spike 上运行
make gen-native_batch_norm      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 378 |
| max\|diff\| | 0 |
| max\|ref\| | 1.84651 |

