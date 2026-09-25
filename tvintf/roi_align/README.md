# roi_align

torchvision 的 `torchvision.ops.roi_align` 在 Hwacha 上的一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/vision/stable/generated/torchvision.ops.roi_align.html

用例：1x4x16x16 特征图、3 个常量 RoI、输出 2x2、spatial_scale 0.5、sampling_ratio 2、aligned=False。

**注意**：这是 torchvision 的 C++ 算子（torch.ops.torchvision.*），torch-mlir 没有 lowering（torchvision 的纯张量 _roi_align 同样 lowering 失败）。导出图按 CUDA 内核的采样规则实现：每个 bin 2x2 个双线性采样点求平均，越界一个像素以内的点夹到边界、更远的读 0，用 one-hot 矩阵做 gather。参考值由真正的 torchvision 调用算出，导出前脚本断言两者一致。

来源：`export_tvi.py` / `intf_ops.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x4x16x16 |
| 输出 | 3x4x2x2 |

## 文件

| 文件 | 内容 |
|---|---|
| `roi_align.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `roi_align_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make roi_align          # 编译 -> roi_align/roi_align.riscv
make roi_align.spike    # 在 Spike 上运行
make gen-roi_align      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 19,626 |
| max\|diff\| | 0 |
| max\|ref\| | 1.07617 |

