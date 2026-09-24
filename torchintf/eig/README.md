# eig

`torch.linalg.eig` 的单函数测试，一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/generated/torch.linalg.eig.html

调用：`torch.linalg.eig(x)`（对称输入（实特征值），输出 [V | Re w]（4x5），按实部升序、符号按第一分量归一）

**注意**：本 case 的导出图与参考不是同一段代码。`torch.linalg` 的分解与求解在 torch-mlir 中没有 lowering（LAPACK 类算子）。导出图是定长的 linalg 组合：循环 Jacobi 旋转（10 遍 x 6 个 (p, q) 对，旋转角用稳定公式 t = 2a_pq·sgn(d) / (|d| + √(d² + 4a_pq²))），特征值用"比它小的个数"做名次的置换矩阵排序，特征向量符号按第一分量归一（参考同样归一）；多个输出用矩阵乘拼接：[A | B] = A @ [I 0] + B @ [0 I]（常量 buffer），不用 tensor.concat；输入对称，参考的复特征值取实部按升序排列、特征向量取实部并归一符号。 `check.bin` 中的参考值仍由真正的 `torch.linalg.eig` 算出，导出前脚本断言两者一致。

来源：`export_intf.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机数据、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 4x4 |
| 输出 | 4x5 |
| 常量 buffer | `E45` 4x5, `E15` 1x5, `I` 4x4, `ar` 4, `below` 4x4, `E1` 4x8, `E2` 4x8 |

## 文件

| 文件 | 内容 |
|---|---|
| `eig.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `eig_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make eig          # 编译 -> eig/eig.riscv
make eig.spike    # 在 Spike 上运行
make gen-eig      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 94,054 |
| max\|diff\| | 6e-06 |
| max\|ref\| | 20.6592 |

