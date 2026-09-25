# five_crop

torchvision 的 `torchvision.transforms.v2.FiveCrop` 在 Hwacha 上的一次应用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.FiveCrop.html

用例：FiveCrop(8)：四角 + 中心，堆叠为 5x3x8x8。

来源：`export_tvi.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机 3x16x16 的 [0, 1] 图像、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 3x16x16 |
| 输出 | 5x3x8x8 |

## 文件

| 文件 | 内容 |
|---|---|
| `five_crop.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `five_crop_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make five_crop          # 编译 -> five_crop/five_crop.riscv
make five_crop.spike    # 在 Spike 上运行
make gen-five_crop      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 347 |
| max\|diff\| | 0 |
| max\|ref\| | 0.999838 |

