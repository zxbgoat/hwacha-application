# fold

`torch.nn.functional.fold`（卷积）的单函数测试，一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.fold.html

导出图计算：`x.reshape(1, 4, 2, 2, 4, 4).permute(0, 1, 4, 2, 5, 3).reshape(1, 4, 8, 8)`

**注意**：本 case 的导出图与参考不是同一段代码。`F.fold` 会 lower 成 `tm_tensor.scatter`（独立 mlir-opt 无法解析）。这里的 2x2、stride 2 无重叠情形是 unfold 的精确逆运算，导出图用 reshape + permute 实现。 `check.bin` 中的参考值仍由真正的 `F.fold` 算出，导出前脚本断言两者一致。

来源：`export_function.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机数据、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x16x16 |
| 输出 | 1x4x8x8 |

## 文件

| 文件 | 内容 |
|---|---|
| `fold.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `fold_check.bin` | 输入与 PyTorch 参考输出（int32 个数 + float32 数组，各一段），host 用 `.incbin` 内嵌 |
| `HWMLIRFLAGS` | 本 case 使用的 hwacha-mlir 映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make fold          # 编译 -> fold/fold.riscv
make fold.spike    # 在 Spike 上运行
make gen-fold      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`。

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 106 |
| max\|diff\| | 0 |
| max\|ref\| | 3.05761 |

