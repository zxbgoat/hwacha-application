# multi_dot

`torch.linalg.multi_dot` 的单函数测试，一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/generated/torch.linalg.multi_dot.html

调用：`torch.linalg.multi_dot([x, y, z])`（y 4x3、z 3x4 常量）

来源：`export_intf.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机数据、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 4x4 |
| 输出 | 4x4 |
| 常量 buffer | `y` 4x3, `z` 3x4 |

## 文件

| 文件 | 内容 |
|---|---|
| `multi_dot.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `multi_dot_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make multi_dot          # 编译 -> multi_dot/multi_dot.riscv
make multi_dot.spike    # 在 Spike 上运行
make gen-multi_dot      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 259 |
| max\|diff\| | 1e-06 |
| max\|ref\| | 21.4514 |

