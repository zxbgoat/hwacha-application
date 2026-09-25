# batched_nms

torchvision 的 `torchvision.ops.batched_nms` 在 Hwacha 上的一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/vision/stable/generated/torchvision.ops.batched_nms.html

用例：8 个框分 3 类，类内 NMS，输出保留掩码。

**注意**：这是 torchvision 的 C++ 算子（torch.ops.torchvision.*），torch-mlir 没有 lowering。导出图：框加上 类别 x 100 的偏移后做同样的贪心 NMS（torchvision 的 offset 技巧）。参考值由真正的 torchvision 调用算出，导出前脚本断言两者一致。

来源：`export_tvi.py` / `intf_ops.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 8x4 |
| 输出 | 8 |

## 文件

| 文件 | 内容 |
|---|---|
| `batched_nms.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `batched_nms_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make batched_nms          # 编译 -> batched_nms/batched_nms.riscv
make batched_nms.spike    # 在 Spike 上运行
make gen-batched_nms      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 2,376 |
| max\|diff\| | 0 |
| max\|ref\| | 1 |

