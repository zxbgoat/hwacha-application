# fftshift

`torch.fft.fftshift` 的单函数测试，一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/generated/torch.fft.fftshift.html

调用：`torch.fft.fftshift(x)`（5x7 张量，两个维度都移位）

**注意**：本 case 的导出图与参考不是同一段代码。`torch.roll` 会 lower 成 slice + concat；导出图用常量下标向量的 `index_select` 逐维做同样的循环移位。 `check.bin` 中的参考值仍由真正的 `torch.fft.fftshift` 算出，导出前脚本断言两者一致。

来源：`export_intf.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机数据、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 5x7 |
| 输出 | 5x7 |
| 常量 buffer | `i0` 5 (int64), `i1` 7 (int64) |

## 文件

| 文件 | 内容 |
|---|---|
| `fftshift.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `fftshift_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make fftshift          # 编译 -> fftshift/fftshift.riscv
make fftshift.spike    # 在 Spike 上运行
make gen-fftshift      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 171 |
| max\|diff\| | 0 |
| max\|ref\| | 2.12918 |

