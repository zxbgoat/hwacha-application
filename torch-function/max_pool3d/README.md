# max_pool3d

`torch.nn.functional.max_pool3d`（池化）的单函数测试，一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.max_pool3d.html

导出图计算：`F.max_pool3d(x, 2)`

来源：`export_function.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机数据、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x4x4x4x4 |
| 输出 | 1x4x2x2x2 |

## 文件

| 文件 | 内容 |
|---|---|
| `max_pool3d.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `max_pool3d_check.bin` | 输入与 PyTorch 参考输出（int32 个数 + float32 数组，各一段），host 用 `.incbin` 内嵌 |
| `HWMLIRFLAGS` | 本 case 使用的 hwacha-mlir 映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make max_pool3d          # 编译 -> max_pool3d/max_pool3d.riscv
make max_pool3d.spike    # 在 Spike 上运行
make gen-max_pool3d      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`。

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 147 |
| max\|diff\| | 0 |
| max\|ref\| | 4.10149 |

