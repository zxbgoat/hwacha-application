# sparse_adam

torch.optim 的 `torch.optim.SparseAdam`（优化算法）在 Hwacha 上的 4 步更新，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/optim.html

问题：参数为 4x4 矩阵与 16 维向量（32 个 float 的输入），loss = 1/2 Σ w (p - t)²（w > 0、t 为常量），梯度 w (p - t) 为闭式；导出图只含更新规则（调用算法的函数式单张量实现 torch.optim.<algo>.<algo>，即 Optimizer.step 所运行的），参考用真正的 Optimizer 在 nn.Parameter 上 backward。

**注意**：稀疏梯度上的 Adam 更新；稠密问题上每一行都被触及，导出图按 _functional.sparse_adam 的公式稠密书写，参考用 grad.to_sparse() 喂真正的 SparseAdam。导出前脚本断言导出体与参考一致。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 32 |
| 输出 | 32 |

## 文件

| 文件 | 内容 |
|---|---|
| `sparse_adam.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `sparse_adam_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make sparse_adam          # 编译 -> sparse_adam/sparse_adam.riscv
make sparse_adam.spike    # 在 Spike 上运行
make gen-sparse_adam      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 5,392 |
| max\|diff\| | 0 |
| max\|ref\| | 2.16786 |

