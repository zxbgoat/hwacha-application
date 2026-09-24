# cross

`torch.linalg.cross` 的单函数测试，一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/generated/torch.linalg.cross.html

调用：`torch.linalg.cross(x, y)`（4 个三维向量与常量 y 的叉积）

**注意**：本 case 的导出图与参考不是同一段代码。`aten.linalg_cross` 的 lowering 要求最后一维为 3 且形状匹配；导出图用常量下标的 `index_select` 写出叉积公式 x[i1]·y[i2] - x[i2]·y[i1]。 `check.bin` 中的参考值仍由真正的 `torch.linalg.cross` 算出，导出前脚本断言两者一致。

来源：`export_intf.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机数据、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 4x3 |
| 输出 | 4x3 |
| 常量 buffer | `y` 4x3, `i1` 3 (int64), `i2` 3 (int64) |

## 文件

| 文件 | 内容 |
|---|---|
| `cross.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `cross_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make cross          # 编译 -> cross/cross.riscv
make cross.spike    # 在 Spike 上运行
make gen-cross      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 434 |
| max\|diff\| | 0 |
| max\|ref\| | 1.8629 |

