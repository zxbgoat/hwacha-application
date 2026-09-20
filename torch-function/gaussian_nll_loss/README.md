# gaussian_nll_loss

`torch.nn.functional.gaussian_nll_loss`（损失函数）的单函数测试，一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.gaussian_nll_loss.html

导出图计算：`(0.5 * (torch.log(var.clamp(min=1e-6)) + (x - t) ** 2 / var.clamp(min=1e-6))).mean().reshape(1)`

**注意**：本 case 的导出图与参考不是同一段代码。`F.gaussian_nll_loss` 对 var 的检查是数据相关的，torch.export 无法追踪。导出图用闭式公式 `0.5 * (log(var) + (x - t)^2 / var)` 的均值。 `check.bin` 中的参考值仍由真正的 `F.gaussian_nll_loss` 算出，导出前脚本断言两者一致。

损失以 mean 归约为标量，作为 1 元素张量返回。

来源：`export_function.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机数据、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 4x8 |
| 输出 | 1 |
| 常量 buffer | `t` 4x8, `var` 4x8 |

## 文件

| 文件 | 内容 |
|---|---|
| `gaussian_nll_loss.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `gaussian_nll_loss_check.bin` | 输入与 PyTorch 参考输出（int32 个数 + float32 数组，各一段），host 用 `.incbin` 内嵌 |
| `HWMLIRFLAGS` | 本 case 使用的 hwacha-mlir 映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make gaussian_nll_loss          # 编译 -> gaussian_nll_loss/gaussian_nll_loss.riscv
make gaussian_nll_loss.spike    # 在 Spike 上运行
make gen-gaussian_nll_loss      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`。

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 643 |
| max\|diff\| | 0 |
| max\|ref\| | 2.54707 |

