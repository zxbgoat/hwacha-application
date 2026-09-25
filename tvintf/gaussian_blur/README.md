# gaussian_blur

torchvision 的 `torchvision.transforms.v2.GaussianBlur` 在 Hwacha 上的一次应用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.GaussianBlur.html

用例：GaussianBlur(5, sigma=1.5)：F.gaussian_blur。

**注意**：GaussianBlur 在 forward 里抽 sigma（数据相关）；固定为 F.gaussian_blur(kernel 5, sigma 1.5)。参考值由真正的 torchvision 调用算出，导出前脚本断言两者一致。

来源：`export_tvi.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机 3x16x16 的 [0, 1] 图像、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 3x16x16 |
| 输出 | 3x16x16 |

## 文件

| 文件 | 内容 |
|---|---|
| `gaussian_blur.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `gaussian_blur_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `hwlib.s` | 库内核 |
| `README.md` | 本文件 |

## 编译与运行

```
make gaussian_blur          # 编译 -> gaussian_blur/gaussian_blur.riscv
make gaussian_blur.spike    # 在 Spike 上运行
make gen-gaussian_blur      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 2,182 |
| max\|diff\| | 0 |
| max\|ref\| | 0.71754 |

