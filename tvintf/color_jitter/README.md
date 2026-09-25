# color_jitter

torchvision 的 `torchvision.transforms.v2.ColorJitter` 在 Hwacha 上的一次应用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.ColorJitter.html

用例：ColorJitter 的一次抽样按类的顺序：brightness 1.3 -> contrast 0.8 -> saturation 1.4 -> hue 0.1。

**注意**：随机变换在 forward 里抽参数、掷是否应用的硬币（即使 p=1、范围退化，对 torch.export 也是数据相关），因此固定为该类实际调用的 functional 算子和一次抽样。F.adjust_hue 的 RGB<->HSV 在 torch-mlir 的 lowering 中失败（arith.cmpi 操作数类型不一致）；导出图用无原地操作、无 aminmax 的同一公式（select 用 one-hot 而不是 gather）重写 rgb_to_hsv / hsv_to_rgb。参考值由真正的 torchvision 调用算出，导出前脚本断言两者一致。

来源：`export_tvi.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机 3x16x16 的 [0, 1] 图像、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 3x16x16 |
| 输出 | 3x16x16 |

## 文件

| 文件 | 内容 |
|---|---|
| `color_jitter.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `color_jitter_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make color_jitter          # 编译 -> color_jitter/color_jitter.riscv
make color_jitter.spike    # 在 Spike 上运行
make gen-color_jitter      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 7,029 |
| max\|diff\| | 0 |
| max\|ref\| | 1 |

