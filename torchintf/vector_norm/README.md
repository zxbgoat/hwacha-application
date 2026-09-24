# vector_norm

`torch.linalg.vector_norm` 的单函数测试，一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/generated/torch.linalg.vector_norm.html

调用：`torch.linalg.vector_norm(x)`（4x4 矩阵全部元素的 2 范数）

**注意**：本 case 的导出图与参考不是同一段代码。导出的图是基于 pow 的归约，hwacha-mlir 不生成内核（no kernels found）；导出图改用 √Σx²。 `check.bin` 中的参考值仍由真正的 `torch.linalg.vector_norm` 算出，导出前脚本断言两者一致。

来源：`export_intf.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机数据、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 4x4 |
| 输出 | 1 |

## 文件

| 文件 | 内容 |
|---|---|
| `vector_norm.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `vector_norm_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make vector_norm          # 编译 -> vector_norm/vector_norm.riscv
make vector_norm.spike    # 在 Spike 上运行
make gen-vector_norm      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 211 |
| max\|diff\| | 0 |
| max\|ref\| | 7.60198 |

