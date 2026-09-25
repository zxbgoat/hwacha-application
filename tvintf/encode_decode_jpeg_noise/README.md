# encode_decode_jpeg_noise

torchvision 的 `torchvision.io.decode_jpeg` 在 Hwacha 上的一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/vision/stable/generated/torchvision.io.decode_jpeg.html

用例：3x16x16 的随机噪声图：decode_jpeg(encode_jpeg(x, quality=75))（最难的输入，仍在 2 级以内）。

**注意**：encode_jpeg / decode_jpeg 是 libjpeg（C++ 算子），没有 lowering。导出图是张量实现的 baseline JPEG 往返：libjpeg 的 16 位定点 RGB->YCbCr、h2v2 色度下采样（2x2 求和加交替的 1/2 偏置后 >> 2）、8x8 DCT 用正交 DCT 矩阵的矩阵乘、quality 75 的标准量化表、系数四舍五入（远离零）、反量化与 IDCT、四舍五入并截断、色度的 "fancy" 三角滤波上采样（3:1 权重加 8/7 偏置）、YCbCr->RGB。libjpeg 用整数近似的 islow DCT，导出图用浮点 DCT，两者相差不超过 2 级（host 容差 2.55）；导出前的断言容差为 2.5。参考值由真正的 torchvision 调用算出，导出前脚本断言两者一致。

来源：`export_tvi.py` / `intf_io.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子；像素以 0..255 的 float 进出。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 3x16x16 |
| 输出 | 3x16x16 |

## 文件

| 文件 | 内容 |
|---|---|
| `encode_decode_jpeg_noise.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `encode_decode_jpeg_noise_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make encode_decode_jpeg_noise          # 编译 -> encode_decode_jpeg_noise/encode_decode_jpeg_noise.riscv
make encode_decode_jpeg_noise.spike    # 在 Spike 上运行
make gen-encode_decode_jpeg_noise      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 9,718 |
| max\|diff\| | 2 |
| max\|ref\| | 255 |

