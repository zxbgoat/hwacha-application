# householder_product

`torch.linalg.householder_product` 的单函数测试，一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/generated/torch.linalg.householder_product.html

调用：`torch.linalg.householder_product(x, tau)`（6x3 的反射向量与常量 tau，输出 H 的前 3 列）

**注意**：本 case 的导出图与参考不是同一段代码。`aten.linalg_householder_product` 没有 lowering。导出图显式相乘 H = Π (I - τ_i v_i v_iᵀ)，v_i 由输入第 i 列取 i 以下的分量并令 v_i[i] = 1。 `check.bin` 中的参考值仍由真正的 `torch.linalg.householder_product` 算出，导出前脚本断言两者一致。

来源：`export_intf.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机数据、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 6x3 |
| 输出 | 6x3 |
| 常量 buffer | `tau` 3, `I6` 6x6, `below6` 3x6, `E63` 6x3 |

## 文件

| 文件 | 内容 |
|---|---|
| `householder_product.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `householder_product_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make householder_product          # 编译 -> householder_product/householder_product.riscv
make householder_product.spike    # 在 Spike 上运行
make gen-householder_product      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 1,723 |
| max\|diff\| | 0 |
| max\|ref\| | 0.840477 |

