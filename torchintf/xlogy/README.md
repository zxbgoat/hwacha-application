# xlogy

`torch.special.xlogy` 的单函数测试，一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/generated/torch.special.xlogy.html

调用：`torch.special.xlogy(x, y)`（y 为常量，x, y ∈ [0.5, 3]）

来源：`export_intf.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机数据、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 4x16 |
| 输出 | 4x16 |
| 常量 buffer | `y` 4x16 |

## 文件

| 文件 | 内容 |
|---|---|
| `xlogy.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `xlogy_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make xlogy          # 编译 -> xlogy/xlogy.riscv
make xlogy.spike    # 在 Spike 上运行
make gen-xlogy      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 363 |
| max\|diff\| | 0 |
| max\|ref\| | 3.06144 |

