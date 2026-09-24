# qr

`torch.linalg.qr` 的单函数测试，一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/generated/torch.linalg.qr.html

调用：`torch.linalg.qr(x)`（输出 [Q | R]（4x8））

**注意**：本 case 的导出图与参考不是同一段代码。改进 Gram–Schmidt（逐列投影、归一，Q = Σ q_j e_jᵀ，R = QᵀA）。多个输出用矩阵乘拼接：[A | B] = A @ [I 0] + B @ [0 I]（常量 buffer），不用 tensor.concat。LAPACK 的 Householder QR 允许 R 对角为负，参考按 sgn(diag R) 把 Q 的列、R 的行归一到 diag(R) > 0。 `check.bin` 中的参考值仍由真正的 `torch.linalg.qr` 算出，导出前脚本断言两者一致。

来源：`export_intf.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机数据、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 4x4 |
| 输出 | 4x8 |
| 常量 buffer | `I` 4x4, `ar` 4, `below` 4x4, `E1` 4x8, `E2` 4x8 |

## 文件

| 文件 | 内容 |
|---|---|
| `qr.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `qr_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make qr          # 编译 -> qr/qr.riscv
make qr.spike    # 在 Spike 上运行
make gen-qr      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 3,319 |
| max\|diff\| | 0 |
| max\|ref\| | 4.28714 |

