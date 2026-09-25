# clip_boxes_to_image

torchvision 的 `torchvision.ops.clip_boxes_to_image` 在 Hwacha 上的一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/vision/stable/generated/torchvision.ops.clip_boxes_to_image.html

用例：裁到 32 x 48 的图像范围。

来源：`export_tvi.py` / `intf_ops.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 8x4 |
| 输出 | 8x4 |

## 文件

| 文件 | 内容 |
|---|---|
| `clip_boxes_to_image.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `clip_boxes_to_image_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make clip_boxes_to_image          # 编译 -> clip_boxes_to_image/clip_boxes_to_image.riscv
make clip_boxes_to_image.spike    # 在 Spike 上运行
make gen-clip_boxes_to_image      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 287 |
| max\|diff\| | 0 |
| max\|ref\| | 48 |

