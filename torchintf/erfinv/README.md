# erfinv

`torch.special.erfinv` 的单函数测试，一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/generated/torch.special.erfinv.html

调用：`torch.special.erfinv(x)`（x ∈ [-0.9, 0.9]）

**注意**：本 case 的导出图与参考不是同一段代码。`aten.erfinv` 的 lowering 失败。导出图用 Giles 的单精度有理逼近（w = -log(1 - x²) 分两段的 9 次多项式）。 `check.bin` 中的参考值仍由真正的 `torch.special.erfinv` 算出，导出前脚本断言两者一致。

来源：`export_intf.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机数据、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 4x16 |
| 输出 | 4x16 |

## 文件

| 文件 | 内容 |
|---|---|
| `erfinv.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `erfinv_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make erfinv          # 编译 -> erfinv/erfinv.riscv
make erfinv.spike    # 在 Spike 上运行
make gen-erfinv      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 1,918 |
| max\|diff\| | 0 |
| max\|ref\| | 1.15925 |

