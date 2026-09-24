# lu_solve

`torch.linalg.lu_solve` 的单函数测试，一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/generated/torch.linalg.lu_solve.html

调用：`torch.linalg.lu_solve(LU, pivots, x)`（输入为右端项 B（4x2），LU 与 pivots 是常量（lu_factor 的结果））

**注意**：本 case 的导出图与参考不是同一段代码。`torch.linalg` 的分解与求解在 torch-mlir 中没有 lowering（LAPACK 类算子）。导出图是定长的 linalg 组合：三角矩阵求逆用幂零级数精确展开：(I + N)⁻¹ = I - N + N² - N³（N 严格三角，N⁴ = 0）；上三角 U = D (I + M)；X = U⁻¹ L⁻¹ B，L、U 取自常量 LU。 `check.bin` 中的参考值仍由真正的 `torch.linalg.lu_solve` 算出，导出前脚本断言两者一致。

来源：`export_intf.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机数据、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 4x2 |
| 输出 | 4x2 |
| 常量 buffer | `LU` 4x4, `piv` 4 (int32), `I` 4x4, `ar` 4, `below` 4x4, `E1` 4x8, `E2` 4x8 |

## 文件

| 文件 | 内容 |
|---|---|
| `lu_solve.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `lu_solve_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make lu_solve          # 编译 -> lu_solve/lu_solve.riscv
make lu_solve.spike    # 在 Spike 上运行
make gen-lu_solve      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 2,013 |
| max\|diff\| | 0 |
| max\|ref\| | 0.316332 |

