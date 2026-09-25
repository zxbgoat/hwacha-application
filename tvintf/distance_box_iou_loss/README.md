# distance_box_iou_loss

torchvision 的 `torchvision.ops.distance_box_iou_loss` 在 Hwacha 上的一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/vision/stable/generated/torchvision.ops.distance_box_iou_loss.html

用例：8 对框的 DIoU 损失之和。

**注意**：同 generalized_box_iou_loss：torch.where 写同一公式。参考值由真正的 torchvision 调用算出，导出前脚本断言两者一致。

来源：`export_tvi.py` / `intf_ops.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 8x4 |
| 输出 | 1 |

## 文件

| 文件 | 内容 |
|---|---|
| `distance_box_iou_loss.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `distance_box_iou_loss_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make distance_box_iou_loss          # 编译 -> distance_box_iou_loss/distance_box_iou_loss.riscv
make distance_box_iou_loss.spike    # 在 Spike 上运行
make gen-distance_box_iou_loss      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 2,492 |
| max\|diff\| | 0 |
| max\|ref\| | 3.26946 |

