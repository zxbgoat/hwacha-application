# nms

torchvision 的 `torchvision.ops.nms` 在 Hwacha 上的一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/vision/stable/generated/torchvision.ops.nms.html

用例：8 个框、常量分数、IoU 阈值 0.5，输出输入顺序下的保留掩码。

**注意**：这是 torchvision 的 C++ 算子（torch.ops.torchvision.*），torch-mlir 没有 lowering。导出图：按分数的名次置换排序，贪心抑制在 8 个框上展开（框 i 被保留当且仅当没有更早保留的框与它 IoU > 阈值）；nms 返回按分数排序的保留下标，参考转成同样的掩码。参考值由真正的 torchvision 调用算出，导出前脚本断言两者一致。

来源：`export_tvi.py` / `intf_ops.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 8x4 |
| 输出 | 8 |

## 文件

| 文件 | 内容 |
|---|---|
| `nms.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `nms_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make nms          # 编译 -> nms/nms.riscv
make nms.spike    # 在 Spike 上运行
make gen-nms      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 2,310 |
| max\|diff\| | 0 |
| max\|ref\| | 1 |

