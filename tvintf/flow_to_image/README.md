# flow_to_image

torchvision 的 `torchvision.utils.flow_to_image` 在 Hwacha 上的一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/vision/stable/generated/torchvision.utils.flow_to_image.html

用例：1x2x8x8 的光流 -> uint8 彩色图。

**注意**：flow_to_image 用 atan2 和整数下标查色轮；导出图按同一算法：按最大范数归一、atan2 由 atan 加象限修正得到、色轮的两个相邻条目用 one-hot 查表后线性插值、按范数向白色淡化、截断。参考值由真正的 torchvision 调用算出，导出前脚本断言两者一致。

来源：`export_tvi.py` / `intf_utils.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x2x8x8 |
| 输出 | 1x3x8x8 |

## 文件

| 文件 | 内容 |
|---|---|
| `flow_to_image.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `flow_to_image_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make flow_to_image          # 编译 -> flow_to_image/flow_to_image.riscv
make flow_to_image.spike    # 在 Spike 上运行
make gen-flow_to_image      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 5,192 |
| max\|diff\| | 0 |
| max\|ref\| | 255 |

