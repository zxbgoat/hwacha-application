# max_unpool1d

`torch.nn.functional.max_unpool1d`（池化）的单函数测试，一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.max_unpool1d.html

导出图计算：`x * (x == F.interpolate(F.max_pool1d(x, 2), scale_factor=2, mode="nearest"))`

**注意**：本 case 的导出图与参考不是同一段代码。max-unpool 的 scatter 是 `tm_tensor` 方言。对同一张量先 max-pool 再 unpool，随机数据无并列极值，结果等于"x 在等于其窗口最大值处保留、其余置零"，导出图用 `x * (x == interpolate(max_pool(x)))` 实现。 `check.bin` 中的参考值仍由真正的 `F.max_unpool1d` 算出，导出前脚本断言两者一致。

来源：`export_function.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机数据、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x4x8 |
| 输出 | 1x4x8 |

## 文件

| 文件 | 内容 |
|---|---|
| `max_unpool1d.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `max_unpool1d_check.bin` | 输入与 PyTorch 参考输出（int32 个数 + float32 数组，各一段），host 用 `.incbin` 内嵌 |
| `HWMLIRFLAGS` | 本 case 使用的 hwacha-mlir 映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make max_unpool1d          # 编译 -> max_unpool1d/max_unpool1d.riscv
make max_unpool1d.spike    # 在 Spike 上运行
make gen-max_unpool1d      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`。

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 609 |
| max\|diff\| | 0 |
| max\|ref\| | 1.85301 |

