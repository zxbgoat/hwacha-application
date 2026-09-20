# embedding_bag

`torch.nn.functional.embedding_bag`（稀疏）的单函数测试，一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.embedding_bag.html

导出图计算：`F.embedding(x.long(), w).mean(1)`

**注意**：本 case 的导出图与参考不是同一段代码。`aten.embedding_bag` 在 torch-mlir 中没有 lowering。单个 bag、mean 模式等于各 embedding 的平均，导出图用 `F.embedding(...).mean(1)` 实现。 `check.bin` 中的参考值仍由真正的 `F.embedding_bag` 算出，导出前脚本断言两者一致。

来源：`export_function.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机数据、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x4 |
| 输出 | 1x16 |
| 常量 buffer | `w` 10x16 |

## 文件

| 文件 | 内容 |
|---|---|
| `embedding_bag.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `embedding_bag_check.bin` | 输入与 PyTorch 参考输出（int32 个数 + float32 数组，各一段），host 用 `.incbin` 内嵌 |
| `HWMLIRFLAGS` | 本 case 使用的 hwacha-mlir 映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make embedding_bag          # 编译 -> embedding_bag/embedding_bag.riscv
make embedding_bag.spike    # 在 Spike 上运行
make gen-embedding_bag      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`。

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 439 |
| max\|diff\| | 0 |
| max\|ref\| | 1.03095 |

