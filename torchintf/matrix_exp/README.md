# matrix_exp

`torch.linalg.matrix_exp` 的单函数测试，一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/generated/torch.linalg.matrix_exp.html

调用：`torch.linalg.matrix_exp(x)`（4x4 矩阵）

**注意**：本 case 的导出图与参考不是同一段代码。`aten.linalg_matrix_exp` 没有 lowering。导出图用缩放平方：exp(A) = (exp(A/4))⁴，exp(A/4) 用 Horner 法则算 15 项 Taylor。 `check.bin` 中的参考值仍由真正的 `torch.linalg.matrix_exp` 算出，导出前脚本断言两者一致。

来源：`export_intf.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机数据、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 4x4 |
| 输出 | 4x4 |
| 常量 buffer | `I` 4x4, `ar` 4, `below` 4x4, `E1` 4x8, `E2` 4x8 |

## 文件

| 文件 | 内容 |
|---|---|
| `matrix_exp.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `matrix_exp_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make matrix_exp          # 编译 -> matrix_exp/matrix_exp.riscv
make matrix_exp.spike    # 在 Spike 上运行
make gen-matrix_exp      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 3,248 |
| max\|diff\| | 0 |
| max\|ref\| | 2.30504 |

