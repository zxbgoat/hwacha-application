# rfft

`torch.fft.rfft` 的单函数测试，一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/generated/torch.fft.rfft.html

调用：`torch.fft.rfft(x)`（实数序列长 16，输出 9 个半谱 bin 的 [实部 | 虚部]）

**注意**：本 case 的导出图与参考不是同一段代码。torch-mlir 没有复数张量：复数张量表示为实数张量，实部与虚部沿最后一维拼接（`[实部 | 虚部]`），实数输入 / 输出保持原形状。沿某一维的 DFT 是与常量块矩阵的矩阵乘：沿最后一维 `[xr | xi] @ [[Fr, Fi], [-Fi, Fr]]`（实数输入 `x @ [Fr | Fi]`）；沿其他维 `Fr @ x + Fi @ (x @ P)`，`P = [[0, I], [-I, 0]]` 把 `[xr | xi]` 变成 `[-xi | xr]`；F = C - iS（正变换）或 (C + iS) / N（逆变换），C、S = cos / sin(2πnk/N)。rfft 取前 N/2+1 列；irfft 是带厄米权重 (1, 2, ..., 2, 1) / N 的 C2R 求和；hfft(x) = N·irfft(conj x)，ihfft(x) = conj(rfft x) / N（n 维同理，N 为各维长度之积）。矩阵在 float64 里生成再转 float32。 `check.bin` 中的参考值仍由真正的 `torch.fft.rfft` 算出，导出前脚本断言两者一致。

来源：`export_intf.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机数据、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 4x16 |
| 输出 | 4x18 |
| 常量 buffer | `B` 16x18 |

## 文件

| 文件 | 内容 |
|---|---|
| `rfft.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `rfft_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make rfft          # 编译 -> rfft/rfft.riscv
make rfft.spike    # 在 Spike 上运行
make gen-rfft      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 274 |
| max\|diff\| | 0 |
| max\|ref\| | 7.25272 |

