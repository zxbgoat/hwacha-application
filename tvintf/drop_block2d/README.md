# drop_block2d

torchvision 的 `torchvision.ops.DropBlock2d` 在 Hwacha 上的一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/vision/stable/generated/torchvision.ops.DropBlock2d.html

用例：DropBlock2d(0.3, 3)，eval 模式：恒等。

来源：`export_tvi.py` / `intf_ops.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x4x16x16 |
| 输出 | 1x4x16x16 |

## 文件

| 文件 | 内容 |
|---|---|
| `drop_block2d.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `drop_block2d_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make drop_block2d          # 编译 -> drop_block2d/drop_block2d.riscv
make drop_block2d.spike    # 在 Spike 上运行
make gen-drop_block2d      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 73 |
| max\|diff\| | 0 |
| max\|ref\| | 3.84602 |

