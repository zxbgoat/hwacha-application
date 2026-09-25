# decode_png_gray

torchvision 的 `torchvision.io.decode_png` 在 Hwacha 上的一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/vision/stable/generated/torchvision.io.decode_png.html

用例：decode_png(encode_png(x), mode=GRAY)。

**注意**：libpng 的 RGB->灰度按 torchvision 配置的 0.2989 / 0.5870 权重以 15 位定点截断计算：floor((9794 R + 19234 G + 3740 B) / 32768)，在随机图上与 decode_png 逐像素拟合确认。参考值由真正的 torchvision 调用算出，导出前脚本断言两者一致。

来源：`export_tvi.py` / `intf_io.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子；像素以 0..255 的 float 进出。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 3x16x16 |
| 输出 | 1x16x16 |

## 文件

| 文件 | 内容 |
|---|---|
| `decode_png_gray.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `decode_png_gray_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make decode_png_gray          # 编译 -> decode_png_gray/decode_png_gray.riscv
make decode_png_gray.spike    # 在 Spike 上运行
make gen-decode_png_gray      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 380 |
| max\|diff\| | 0 |
| max\|ref\| | 240 |

