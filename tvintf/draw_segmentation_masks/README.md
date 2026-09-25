# draw_segmentation_masks

torchvision 的 `torchvision.utils.draw_segmentation_masks` 在 Hwacha 上的一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/vision/stable/generated/torchvision.utils.draw_segmentation_masks.html

用例：2 个重叠的掩码、alpha 0.6，两种颜色。

**注意**：torchvision 通过布尔索引赋值（无 lowering）；导出图按同一规则：每个掩码涂色、重叠像素置 0、再与原图按 alpha 混合并截断。参考值由真正的 torchvision 调用算出，导出前脚本断言两者一致。

来源：`export_tvi.py` / `intf_utils.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 3x16x16 |
| 输出 | 3x16x16 |

## 文件

| 文件 | 内容 |
|---|---|
| `draw_segmentation_masks.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `draw_segmentation_masks_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make draw_segmentation_masks          # 编译 -> draw_segmentation_masks/draw_segmentation_masks.riscv
make draw_segmentation_masks.spike    # 在 Spike 上运行
make gen-draw_segmentation_masks      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 1,120 |
| max\|diff\| | 0 |
| max\|ref\| | 254 |

