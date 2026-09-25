# random_photometric_distort

torchvision 的 `torchvision.transforms.v2.RandomPhotometricDistort` 在 Hwacha 上的一次应用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomPhotometricDistort.html

用例：RandomPhotometricDistort 的一次抽样：brightness 0.9 -> contrast 1.3 -> saturation 0.7 -> hue -0.05。

**注意**：随机变换在 forward 里抽参数、掷是否应用的硬币（即使 p=1、范围退化，对 torch.export 也是数据相关），因此固定为该类实际调用的 functional 算子和一次抽样。adjust_hue 同 color_jitter 的重写。参考值由真正的 torchvision 调用算出，导出前脚本断言两者一致。

来源：`export_tvi.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机 3x16x16 的 [0, 1] 图像、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 3x16x16 |
| 输出 | 3x16x16 |

## 文件

| 文件 | 内容 |
|---|---|
| `random_photometric_distort.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `random_photometric_distort_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make random_photometric_distort          # 编译 -> random_photometric_distort/random_photometric_distort.riscv
make random_photometric_distort.spike    # 在 Spike 上运行
make gen-random_photometric_distort      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 7,029 |
| max\|diff\| | 0 |
| max\|ref\| | 0.997018 |

