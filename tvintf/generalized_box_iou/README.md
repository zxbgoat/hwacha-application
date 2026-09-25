# generalized_box_iou

torchvision 的 `torchvision.ops.generalized_box_iou` 在 Hwacha 上的一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/vision/stable/generated/torchvision.ops.generalized_box_iou.html

用例：8 x 8 的 GIoU 矩阵。

来源：`export_tvi.py` / `intf_ops.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 8x4 |
| 输出 | 8x8 |

## 文件

| 文件 | 内容 |
|---|---|
| `generalized_box_iou.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `generalized_box_iou_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make generalized_box_iou          # 编译 -> generalized_box_iou/generalized_box_iou.riscv
make generalized_box_iou.spike    # 在 Spike 上运行
make gen-generalized_box_iou      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 1,248 |
| max\|diff\| | 0 |
| max\|ref\| | 0.954893 |

