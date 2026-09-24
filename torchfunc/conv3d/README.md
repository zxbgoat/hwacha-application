# conv3d

`torch.nn.functional.conv3d`（卷积）的单函数测试，一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.conv3d.html

导出图计算：`F.conv3d(x, w, b, padding=1)`

来源：`export_function.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机数据、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x4x4x4x4 |
| 输出 | 1x8x4x4x4 |
| 常量 buffer | `w` 8x4x3x3x3, `b` 8 |

## 文件

| 文件 | 内容 |
|---|---|
| `conv3d.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `conv3d_check.bin` | 输入与 PyTorch 参考输出（int32 个数 + float32 数组，各一段），host 用 `.incbin` 内嵌 |
| `HWMLIRFLAGS` | 本 case 使用的 hwacha-mlir 映射选项 |
| `hwlib.s` | 卷积 / 池化库内核（本 case 的汇编调用了它） |
| `README.md` | 本文件 |

## 编译与运行

```
make conv3d          # 编译 -> conv3d/conv3d.riscv
make conv3d.spike    # 在 Spike 上运行
make gen-conv3d      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`。

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 17,661 |
| max\|diff\| | 6e-06 |
| max\|ref\| | 30.5458 |

