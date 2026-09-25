# write_png_read_image

torchvision 的 `torchvision.io.read_image` 在 Hwacha 上的一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/vision/stable/generated/torchvision.io.read_image.html

用例：write_png 写到临时文件后 read_image(mode=GRAY) 读回。

**注意**：同 decode_png_gray 的灰度转换；文件读写本身没有张量计算。参考值由真正的 torchvision 调用算出，导出前脚本断言两者一致。

来源：`export_tvi.py` / `intf_io.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子；像素以 0..255 的 float 进出。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 3x16x16 |
| 输出 | 1x16x16 |

## 文件

| 文件 | 内容 |
|---|---|
| `write_png_read_image.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `write_png_read_image_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make write_png_read_image          # 编译 -> write_png_read_image/write_png_read_image.riscv
make write_png_read_image.spike    # 在 Spike 上运行
make gen-write_png_read_image      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 380 |
| max\|diff\| | 0 |
| max\|ref\| | 240 |

