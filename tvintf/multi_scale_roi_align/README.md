# multi_scale_roi_align

torchvision 的 `torchvision.ops.MultiScaleRoIAlign` 在 Hwacha 上的一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/vision/stable/generated/torchvision.ops.MultiScaleRoIAlign.html

用例：单一层级 ['0']、输出 2、sampling_ratio 2，64x64 图像上推断 scale = 1/4。

**注意**：这是 torchvision 的 C++ 算子（torch.ops.torchvision.*），torch-mlir 没有 lowering。单层级下等于 roi_align(scale 1/4, aligned=False)，同 roi_align 的实现。参考值由真正的 torchvision 调用算出，导出前脚本断言两者一致。

来源：`export_tvi.py` / `intf_ops.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x4x16x16 |
| 输出 | 3x4x2x2 |

## 文件

| 文件 | 内容 |
|---|---|
| `multi_scale_roi_align.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `multi_scale_roi_align_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make multi_scale_roi_align          # 编译 -> multi_scale_roi_align/multi_scale_roi_align.riscv
make multi_scale_roi_align.spike    # 在 Spike 上运行
make gen-multi_scale_roi_align      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 19,627 |
| max\|diff\| | 0 |
| max\|ref\| | 0.955542 |

