# save_image

torchvision 的 `torchvision.utils.save_image` 在 Hwacha 上的一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/vision/stable/generated/torchvision.utils.save_image.html

用例：save_image 写入 PNG 的像素：make_grid 后 x·255 + 0.5、clamp、截断为 uint8。

**注意**：参考值是真正 save_image 写出的 PNG 读回来的像素；导出图算写入前的同一量化。参考值由真正的 torchvision 调用算出，导出前脚本断言两者一致。

来源：`export_tvi.py` / `intf_utils.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 4x3x8x8 |
| 输出 | 3x19x19 |

## 文件

| 文件 | 内容 |
|---|---|
| `save_image.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `save_image_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make save_image          # 编译 -> save_image/save_image.riscv
make save_image.spike    # 在 Spike 上运行
make gen-save_image      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 834 |
| max\|diff\| | 0 |
| max\|ref\| | 255 |

