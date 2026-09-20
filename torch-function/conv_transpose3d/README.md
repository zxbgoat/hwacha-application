# conv_transpose3d

`torch.nn.functional.conv_transpose3d`（卷积）的单函数测试，一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.conv_transpose3d.html

导出图计算：`F.conv_transpose3d(x, w, b, stride=2)`

来源：`export_function.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机数据、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x4x4x4x4 |
| 输出 | 1x8x8x8x8 |
| 常量 buffer | `w` 4x8x2x2x2, `b` 8 |

## 文件

| 文件 | 内容 |
|---|---|
| `conv_transpose3d.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `conv_transpose3d_check.bin` | 输入与 PyTorch 参考输出（int32 个数 + float32 数组，各一段），host 用 `.incbin` 内嵌 |
| `HWMLIRFLAGS` | 本 case 使用的 hwacha-mlir 映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make conv_transpose3d          # 编译 -> conv_transpose3d/conv_transpose3d.riscv
make conv_transpose3d.spike    # 在 Spike 上运行
make gen-conv_transpose3d      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all --unroll-small=2`。

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 5,355 |
| max\|diff\| | 1e-06 |
| max\|ref\| | 10.6495 |

