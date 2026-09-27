# averaged_model_ema

torch.optim 的 `torch.optim.swa_utils.AveragedModel(avg_fn=get_ema_avg_fn(0.9))`（权重平均）在 Hwacha 上的 8 步更新，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/optim.html

问题：参数为 4x4 矩阵与 16 维向量（32 个 float 的输入），loss = 1/2 Σ w (p - t)²（w > 0、t 为常量），梯度 w (p - t) 为闭式；导出图只含更新规则；SGD 8 步，每步后 AveragedModel.update_parameters（第一次为复制），输出 [参数 | 平均参数]，参考用真正的 AveragedModel。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 32 |
| 输出 | 64 |

## 文件

| 文件 | 内容 |
|---|---|
| `averaged_model_ema.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `averaged_model_ema_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make averaged_model_ema          # 编译 -> averaged_model_ema/averaged_model_ema.riscv
make averaged_model_ema.spike    # 在 Spike 上运行
make gen-averaged_model_ema      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 4,990 |
| max\|diff\| | 0 |
| max\|ref\| | 1.97528 |

