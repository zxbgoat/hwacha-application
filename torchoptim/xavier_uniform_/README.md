# xavier_uniform_

torch 的 `torch.nn.init.xavier_uniform_(gain=1.5)`（nn.init）在 Hwacha 上的一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/nn.init.html

初始化器就地填充张量；用例在输入副本上运行并返回填充结果。随机初始化器在 torch.manual_seed(0) 下抽样（RNG 算子无 lowering / Hwacha 上不复现 CPU 生成器），seed 0 的抽样为用例常量，参考重新抽取；dirac_ / eye_ 的 delta 模式为常量；orthogonal_ 为固定高斯的 Gram-Schmidt（linalg.qr 无 lowering），列符号按 sign(diag(R)) 修正。

；SGD 8 步，每步后 AveragedModel.update_parameters（第一次为复制），输出 [参数 | 平均参数]，参考用真正的 AveragedModel。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 4x8 |
| 输出 | 4x8 |

## 文件

| 文件 | 内容 |
|---|---|
| `xavier_uniform_.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `xavier_uniform__check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make xavier_uniform_          # 编译 -> xavier_uniform_/xavier_uniform_.riscv
make xavier_uniform_.spike    # 在 Spike 上运行
make gen-xavier_uniform_      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 121 |
| max\|diff\| | 0 |
| max\|ref\| | 1.0133 |

