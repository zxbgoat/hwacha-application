# adam_amsgrad

torch.optim 的 `torch.optim.Adam(amsgrad=True, weight_decay=0.01)`（优化算法）在 Hwacha 上的 4 步更新，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/optim.html

问题：参数为 4x4 矩阵与 16 维向量（32 个 float 的输入），loss = 1/2 Σ w (p - t)²（w > 0、t 为常量），梯度 w (p - t) 为闭式；导出图只含更新规则（调用算法的函数式单张量实现 torch.optim.<algo>.<algo>，即 Optimizer.step 所运行的），参考用真正的 Optimizer 在 nn.Parameter 上 backward。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 32 |
| 输出 | 32 |

## 文件

| 文件 | 内容 |
|---|---|
| `adam_amsgrad.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `adam_amsgrad_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make adam_amsgrad          # 编译 -> adam_amsgrad/adam_amsgrad.riscv
make adam_amsgrad.spike    # 在 Spike 上运行
make gen-adam_amsgrad      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 7,286 |
| max\|diff\| | 3e-06 |
| max\|ref\| | 2.16787 |

