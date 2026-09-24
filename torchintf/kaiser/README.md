# kaiser

`torch.signal.windows.kaiser` 的单函数测试，一次调用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/docs/2.14/generated/torch.signal.windows.kaiser.html

调用：`x * torch.signal.windows.kaiser(16, beta=12.0)`（给 4 行、长 16 的信号加窗；窗函数没有张量输入，窗在导出图里按定义计算（cos / sin / exp / pow / abs 在 Hwacha 上求值））

**注意**：本 case 的导出图与参考不是同一段代码。`torch.i0`（零阶修正贝塞尔函数）在 torch-mlir 中没有 lowering。导出图用幂级数 I0(z) = Σ_k (z/2)^{2k} / (k!)^2 按 Horner 法则算 30 项（beta = 12 时 z/2 <= 6，float32 下与 torch.i0 相差 < 1e-6），分母 I0(beta) 是常量。 `check.bin` 中的参考值仍由真正的 `torch.signal.windows.kaiser` 算出，导出前脚本断言两者一致。

来源：`export_intf.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机数据、固定种子。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 4x16 |
| 输出 | 4x16 |
| 常量 buffer | `n` 16 |

## 文件

| 文件 | 内容 |
|---|---|
| `kaiser.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `kaiser_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make kaiser          # 编译 -> kaiser/kaiser.riscv
make kaiser.spike    # 在 Spike 上运行
make gen-kaiser      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 3,781 |
| max\|diff\| | 0 |
| max\|ref\| | 2.08242 |

