# make_grid_normalize

torchvision 的 `torchvision.utils.make_grid` 在 Hwacha 上的一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/vision/stable/generated/torchvision.utils.make_grid.html

用例：normalize=True、scale_each=True：每张图按自身 min / max 归一后拼网格。

**注意**：normalize 把张量自身的 min / max 取成 python 数做 clamp 边界（数据相关）；导出图用张量运算做同样的逐图归一化再 make_grid。参考值由真正的 torchvision 调用算出，导出前脚本断言两者一致。

来源：`export_tvi.py` / `intf_utils.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 4x3x8x8 |
| 输出 | 3x19x19 |

## 文件

| 文件 | 内容 |
|---|---|
| `make_grid_normalize.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `make_grid_normalize_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make make_grid_normalize          # 编译 -> make_grid_normalize/make_grid_normalize.riscv
make make_grid_normalize.spike    # 在 Spike 上运行
make gen-make_grid_normalize      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 1,931 |
| max\|diff\| | 3e-06 |
| max\|ref\| | 1 |

