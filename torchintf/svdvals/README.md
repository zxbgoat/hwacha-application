# svdvals

`torch.linalg.svdvals` 的单函数测试，一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/generated/torch.linalg.svdvals.html

调用：`torch.linalg.svdvals(x)`（降序奇异值）

**注意**：本 case 的导出图与参考不是同一段代码。`torch.linalg` 的分解与求解在 torch-mlir 中没有 lowering（LAPACK 类算子）。导出图是定长的 linalg 组合：循环 Jacobi 旋转（10 遍 x 6 个 (p, q) 对，旋转角用稳定公式 t = 2a_pq·sgn(d) / (|d| + √(d² + 4a_pq²))），特征值用"比它小的个数"做名次的置换矩阵排序，特征向量符号按第一分量归一（参考同样归一）；Jacobi 求 AᵀA 的特征分解得 V 与 σ = √w（降序），U = A V / σ；参考 svd 的 V 列符号同样按第一分量归一、U 随之。 `check.bin` 中的参考值仍由真正的 `torch.linalg.svdvals` 算出，导出前脚本断言两者一致。

来源：`export_intf.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机数据、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 4x4 |
| 输出 | 4 |
| 常量 buffer | `I` 4x4, `ar` 4, `below` 4x4, `E1` 4x8, `E2` 4x8 |

## 文件

| 文件 | 内容 |
|---|---|
| `svdvals.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `svdvals_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make svdvals          # 编译 -> svdvals/svdvals.riscv
make svdvals.spike    # 在 Spike 上运行
make gen-svdvals      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 86,076 |
| max\|diff\| | 1e-06 |
| max\|ref\| | 4.54524 |

