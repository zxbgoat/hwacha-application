# random_autocontrast

torchvision 的 `torchvision.transforms.v2.RandomAutocontrast` 在 Hwacha 上的一次应用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomAutocontrast.html

用例：F.autocontrast。

**注意**：随机变换在 forward 里抽参数、掷是否应用的硬币（即使 p=1、范围退化，对 torch.export 也是数据相关），因此固定为该类实际调用的 functional 算子和一次抽样。F.autocontrast 的导出含 tm_tensor.scan / scatter（索引赋值）；导出图用同一公式 (x - min) / (max - min)、torch.where 处理 max == min。参考值由真正的 torchvision 调用算出，导出前脚本断言两者一致。

来源：`export_tvi.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机 3x16x16 的 [0, 1] 图像、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 3x16x16 |
| 输出 | 3x16x16 |

## 文件

| 文件 | 内容 |
|---|---|
| `random_autocontrast.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `random_autocontrast_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make random_autocontrast          # 编译 -> random_autocontrast/random_autocontrast.riscv
make random_autocontrast.spike    # 在 Spike 上运行
make gen-random_autocontrast      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 1,511 |
| max\|diff\| | 0 |
| max\|ref\| | 1 |

