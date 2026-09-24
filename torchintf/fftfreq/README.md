# fftfreq

`torch.fft.fftfreq` 的单函数测试，一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/generated/torch.fft.fftfreq.html

调用：`x + torch.fft.fftfreq(16)`（频率向量是常量，加到长 16 的输入上）

**注意**：本 case 的导出图与参考不是同一段代码。`aten.fft_fftfreq` / `aten.fft_rfftfreq` 在 torch-mlir 中没有 lowering。频率向量本就是常量，导出图把它作为 buffer 加到输入上。 `check.bin` 中的参考值仍由真正的 `torch.fft.fftfreq` 算出，导出前脚本断言两者一致。

来源：`export_intf.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机数据、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 16 |
| 输出 | 16 |
| 常量 buffer | `f` 16 |

## 文件

| 文件 | 内容 |
|---|---|
| `fftfreq.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `fftfreq_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make fftfreq          # 编译 -> fftfreq/fftfreq.riscv
make fftfreq.spike    # 在 Spike 上运行
make gen-fftfreq      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 76 |
| max\|diff\| | 0 |
| max\|ref\| | 1.81034 |

