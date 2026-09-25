# conv3d_norm_activation

torchvision 的 `torchvision.ops.Conv3dNormActivation` 在 Hwacha 上的一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/vision/stable/generated/torchvision.ops.Conv3dNormActivation.html

用例：Conv3dNormActivation(2, 4, 3)。

来源：`export_tvi.py` / `intf_ops.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x2x6x6x6 |
| 输出 | 1x4x6x6x6 |

## 文件

| 文件 | 内容 |
|---|---|
| `conv3d_norm_activation.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `conv3d_norm_activation_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `hwlib.s` | 库内核 |
| `README.md` | 本文件 |

## 编译与运行

```
make conv3d_norm_activation          # 编译 -> conv3d_norm_activation/conv3d_norm_activation.riscv
make conv3d_norm_activation.spike    # 在 Spike 上运行
make gen-conv3d_norm_activation      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 9,792 |
| max\|diff\| | 0 |
| max\|ref\| | 1.65652 |

