# forward_kl_loss

torchtune 的 `torchtune.modules.loss.ForwardKLLoss` 在 Hwacha 上的一次前向，与 PyTorch 逐元素比对。文档：https://meta-pytorch.org/torchtune/0.6/generated/torchtune.modules.loss.ForwardKLLoss.html

用例：学生 / 教师 logits (1, 8, 16) 的前向 KL，标签含一个 ignore_index。

**注意**：损失对未掩码 token 数做数据相关分支（if sum_masks == 0），torch.export 无法追踪；导出图写出公式 -Σ_i m_i Σ_v p_t log p_s / Σ_i m_i。参考值由真正的 torchtune 调用算出，导出前脚本断言两者一致。

来源：`export_ttf.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机权重与输入、固定种子；规模很小（embed 32、4 头 x head_dim 8、序列 8、词表 16）。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x8x16 |
| 输出 | 1 |

## 文件

| 文件 | 内容 |
|---|---|
| `forward_kl_loss.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `forward_kl_loss_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make forward_kl_loss          # 编译 -> forward_kl_loss/forward_kl_loss.riscv
make forward_kl_loss.spike    # 在 Spike 上运行
make gen-forward_kl_loss      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 2,016 |
| max\|diff\| | 0 |
| max\|ref\| | 3.23672 |

