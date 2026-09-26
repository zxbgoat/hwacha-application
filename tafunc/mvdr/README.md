# mvdr

torchaudio 的 `torchaudio.transforms.MVDR` 在 Hwacha 上的一次应用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.MVDR.html

用例：MVDR(ref_channel 0, solution 'ref_channel')：PSD -> Souden 权重 -> 波束形成，输出增强谱 (17, 17, 2)。

**注意**：MVDR 在 cdouble 下用 linalg.solve；导出图用实数算术：2x2 复矩阵求逆闭式、对角加载、迹归一、conj(w)·X，容差 2e-3。参考值由真正的 torchaudio 变换算出，导出前脚本断言两者一致。

来源：`export_taf.py` / `taf_lib.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子；复数谱以末维为 (re, im) 的实张量进出。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 2x17x17x2 |
| 输出 | 17x17x2 |

## 文件

| 文件 | 内容 |
|---|---|
| `mvdr.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `mvdr_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make mvdr          # 编译 -> mvdr/mvdr.riscv
make mvdr.spike    # 在 Spike 上运行
make gen-mvdr      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 13,397 |
| max\|diff\| | 0 |
| max\|ref\| | 8.78615 |

