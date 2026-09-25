# draw_bounding_boxes

torchvision 的 `torchvision.utils.draw_bounding_boxes` 在 Hwacha 上的一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/vision/stable/generated/torchvision.utils.draw_bounding_boxes.html

用例：16x16 uint8 图上画 3 个 width 2 的彩色框（后画的覆盖先画的）。

**注意**：draw_bounding_boxes 用 PIL 光栅化；导出图在坐标网格上按 PIL 的矩形轮廓规则（外框含、内缩 width 的框不含）逐框覆盖，与 PIL 逐像素一致。参考值由真正的 torchvision 调用算出，导出前脚本断言两者一致。

来源：`export_tvi.py` / `intf_utils.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 3x16x16 |
| 输出 | 3x16x16 |

## 文件

| 文件 | 内容 |
|---|---|
| `draw_bounding_boxes.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `draw_bounding_boxes_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make draw_bounding_boxes          # 编译 -> draw_bounding_boxes/draw_bounding_boxes.riscv
make draw_bounding_boxes.spike    # 在 Spike 上运行
make gen-draw_bounding_boxes      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 3,420 |
| max\|diff\| | 0 |
| max\|ref\| | 255 |

