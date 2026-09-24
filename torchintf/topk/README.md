# topk

`torch.topk` 的单函数测试，一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/generated/torch.topk.html

调用：`torch.topk(x, 4)`（dim=-1、largest=True、sorted=True：每行最大的 4 个值（降序）及其下标，输出为每行 [values | indices]）

**注意**：本 case 的导出图与参考不是同一段代码。`aten.topk` 在 torch-mlir 中 lower 成 `tm_tensor.sort`，hwacha-mlir 不接受（与 `../torchfunc` 的 fold / max_unpool 同一限制）。导出图用等价的 linalg 组合：每个元素的名次 = 本行中严格大于它的元素个数（随机数据无并列），名次为 i 的元素用 one-hot 求和选出：`values_i = sum_j x_j [rank_j == i]`，`indices_i = sum_j j [rank_j == i]`。每行 O(N^2) 次比较而不是排序，值与下标都精确。 `check.bin` 中的参考值仍由真正的 `torch.topk` 算出，导出前脚本断言两者一致。

来源：`export_intf.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机数据、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 4x16 |
| 输出 | 4x8 |
| 常量 buffer | `r` 8, `isval` 8, `idx` 16 |

## 文件

| 文件 | 内容 |
|---|---|
| `topk.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `topk_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make topk          # 编译 -> topk/topk.riscv
make topk.spike    # 在 Spike 上运行
make gen-topk      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 1,116 |
| max\|diff\| | 0 |
| max\|ref\| | 14 |

