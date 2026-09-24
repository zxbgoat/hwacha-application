# bessel_y0

`torch.special.bessel_y0` 的单函数测试，一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/generated/torch.special.bessel_y0.html

调用：`torch.special.bessel_y0(x)`（x ∈ [0.5, 4]）

**注意**：本 case 的导出图与参考不是同一段代码。`torch.special` 的这个函数在 torch-mlir 中没有 lowering。导出图用对数级数 (2/π)[(ln(x/2) + γ) J₀ + Σ (-1)ᵏ⁺¹ H_k tᵏ/(k!)²]。 `check.bin` 中的参考值仍由真正的 `torch.special.bessel_y0` 算出，导出前脚本断言两者一致。

来源：`export_intf.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机数据、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 4x16 |
| 输出 | 4x16 |

## 文件

| 文件 | 内容 |
|---|---|
| `bessel_y0.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `bessel_y0_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make bessel_y0          # 编译 -> bessel_y0/bessel_y0.riscv
make bessel_y0.spike    # 在 Spike 上运行
make gen-bessel_y0      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 4,468 |
| max\|diff\| | 0 |
| max\|ref\| | 0.520622 |

