# lazybatchnorm2d

`torch.nn` 单层测试：**rand_bn(m.eval())（其中 `m=nn.LazyBatchNorm2d(); m(x4)`）**，一次前向，与 PyTorch 逐元素比对。

输入：`x4`（`x1`/`x4`/`x8`/`x3` 为 1x4x8 / 1x4x8x8 / 1x8x8x8 / 1x4x4x4x4 的随机张量）。

包装说明：`rand_bn`：给 BatchNorm 随机的 running 统计量与仿射参数，避免退化。

来源：`hwacha-cc/test/modules/export_module.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机权重、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x4x8x8 |
| 输出 | 1x4x8x8 |
| 参数量 | 8 |
| 常量 buffer | `running_mean` 4, `running_var` 4, `num_batches_tracked` 标量 (int64) |

## 文件

| 文件 | 内容 |
|---|---|
| `lazybatchnorm2d.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `lazybatchnorm2d_check.bin` | 输入与 PyTorch 参考输出（int32 个数 + float32 数组，各一段），host 用 `.incbin` 内嵌 |
| `README.md` | 本文件 |

## 编译与运行

```
make lazybatchnorm2d          # 编译 -> lazybatchnorm2d/lazybatchnorm2d.riscv
make lazybatchnorm2d.spike    # 在 Spike 上运行
```

hwacha-mlir 映射：`--collapse-all`（所有并行维映射到 lane）。

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 367 |
| max\|diff\| | 0 |
| max\|ref\| | 4.50465 |

