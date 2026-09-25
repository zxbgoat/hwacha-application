# ps_roi_align_module

torchvision 的 `torchvision.ops.PSRoIAlign` 在 Hwacha 上的一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/vision/stable/generated/torchvision.ops.PSRoIAlign.html

用例：PSRoIAlign(2, 1.0, 2) 模块。

**注意**：同 ps_roi_align 的实现。参考值由真正的 torchvision 调用算出，导出前脚本断言两者一致。

来源：`export_tvi.py` / `intf_ops.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x16x16x16 |
| 输出 | 3x4x2x2 |

## 文件

| 文件 | 内容 |
|---|---|
| `ps_roi_align_module.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `ps_roi_align_module_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make ps_roi_align_module          # 编译 -> ps_roi_align_module/ps_roi_align_module.riscv
make ps_roi_align_module.spike    # 在 Spike 上运行
make gen-ps_roi_align_module      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 39,555 |
| max\|diff\| | 0 |
| max\|ref\| | 0.770732 |

