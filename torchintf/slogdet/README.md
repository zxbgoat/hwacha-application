# slogdet

`torch.linalg.slogdet` 的单函数测试，一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/generated/torch.linalg.slogdet.html

调用：`torch.linalg.slogdet(x)`（输出 [sign, logabsdet]）

**注意**：本 case 的导出图与参考不是同一段代码。`torch.linalg` 的分解与求解在 torch-mlir 中没有 lowering（LAPACK 类算子）。导出图是定长的 linalg 组合：无主元的 Gauss 消元（三个 Gauss 变换 I - l_k e_kᵀ，掩码取列 k 主元下方）；输入对角占优，LAPACK 也不选主元（导出时断言 pivots 为单位置换）；sign = Π sgn(u_ii)、logabsdet = Σ log|u_ii|，两个标量经 one-hot 拼成长 2 的向量。 `check.bin` 中的参考值仍由真正的 `torch.linalg.slogdet` 算出，导出前脚本断言两者一致。

来源：`export_intf.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机数据、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 4x4 |
| 输出 | 2 |
| 常量 buffer | `I` 4x4, `ar` 4, `below` 4x4, `E1` 4x8, `E2` 4x8 |

## 文件

| 文件 | 内容 |
|---|---|
| `slogdet.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `slogdet_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make slogdet          # 编译 -> slogdet/slogdet.riscv
make slogdet.spike    # 在 Spike 上运行
make gen-slogdet      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 2,363 |
| max\|diff\| | 0 |
| max\|ref\| | 5.24038 |

