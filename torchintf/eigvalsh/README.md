# eigvalsh

`torch.linalg.eigvalsh` 的单函数测试，一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/generated/torch.linalg.eigvalsh.html

调用：`torch.linalg.eigvalsh(x)`（升序特征值）

**注意**：本 case 的导出图与参考不是同一段代码。`torch.linalg` 的分解与求解在 torch-mlir 中没有 lowering（LAPACK 类算子）。导出图是定长的 linalg 组合：循环 Jacobi 旋转（10 遍 x 6 个 (p, q) 对，旋转角用稳定公式 t = 2a_pq·sgn(d) / (|d| + √(d² + 4a_pq²))），特征值用"比它小的个数"做名次的置换矩阵排序，特征向量符号按第一分量归一（参考同样归一）。 `check.bin` 中的参考值仍由真正的 `torch.linalg.eigvalsh` 算出，导出前脚本断言两者一致。

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
| `eigvalsh.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `eigvalsh_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make eigvalsh          # 编译 -> eigvalsh/eigvalsh.riscv
make eigvalsh.spike    # 在 Spike 上运行
make gen-eigvalsh      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 85,971 |
| max\|diff\| | 5e-06 |
| max\|ref\| | 20.6592 |

