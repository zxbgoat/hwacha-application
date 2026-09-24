# tensorinv

`torch.linalg.tensorinv` 的单函数测试，一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/generated/torch.linalg.tensorinv.html

调用：`torch.linalg.tensorinv(x, ind=1)`（输入 4x2x2，输出 2x2x4）

**注意**：本 case 的导出图与参考不是同一段代码。`torch.linalg` 的分解与求解在 torch-mlir 中没有 lowering（LAPACK 类算子）。导出图是定长的 linalg 组合：Newton–Schulz 迭代 X <- X (2I - A X)，X0 = Aᵀ / (‖A‖₁‖A‖∞)，30 次，每次两个 4x4 矩阵乘；先 reshape 成 4x4 求逆再 reshape 回 2x2x4。 `check.bin` 中的参考值仍由真正的 `torch.linalg.tensorinv` 算出，导出前脚本断言两者一致。

来源：`export_intf.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机数据、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 4x2x2 |
| 输出 | 2x2x4 |
| 常量 buffer | `I` 4x4, `ar` 4, `below` 4x4, `E1` 4x8, `E2` 4x8 |

## 文件

| 文件 | 内容 |
|---|---|
| `tensorinv.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `tensorinv_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make tensorinv          # 编译 -> tensorinv/tensorinv.riscv
make tensorinv.spike    # 在 Spike 上运行
make gen-tensorinv      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 9,049 |
| max\|diff\| | 0 |
| max\|ref\| | 0.314498 |

