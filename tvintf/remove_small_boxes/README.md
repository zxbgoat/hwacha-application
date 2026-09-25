# remove_small_boxes

torchvision 的 `torchvision.ops.remove_small_boxes` 在 Hwacha 上的一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/vision/stable/generated/torchvision.ops.remove_small_boxes.html

用例：min_size 8，输出保留掩码（0/1 向量）。

**注意**：remove_small_boxes 返回下标（长度数据相关）；导出图输出等价的保留掩码，参考把下标转成掩码比对。参考值由真正的 torchvision 调用算出，导出前脚本断言两者一致。

来源：`export_tvi.py` / `intf_ops.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 8x4 |
| 输出 | 8 |

## 文件

| 文件 | 内容 |
|---|---|
| `remove_small_boxes.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `remove_small_boxes_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make remove_small_boxes          # 编译 -> remove_small_boxes/remove_small_boxes.riscv
make remove_small_boxes.spike    # 在 Spike 上运行
make gen-remove_small_boxes      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 353 |
| max\|diff\| | 0 |
| max\|ref\| | 1 |

