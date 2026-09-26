# inverse_mel_scale

torchaudio 的 `torchaudio.transforms.InverseMelScale` 在 Hwacha 上的一次应用，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.InverseMelScale.html

用例：InverseMelScale(n_stft 17, 8 个 mel 带)：从 mel 谱最小二乘反解线性谱。

**注意**：aten.linalg_lstsq 没有 lowering；driver gels 给欠定系统的最小范数解，导出图用常量伪逆 Aᵀ(AAᵀ)⁻¹（构造时断言与 lstsq 一致）再 relu。参考值由真正的 torchaudio 变换算出，导出前脚本断言两者一致。

来源：`export_taf.py` / `taf_lib.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子；复数谱以末维为 (re, im) 的实张量进出。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x8x17 |
| 输出 | 1x17x17 |

## 文件

| 文件 | 内容 |
|---|---|
| `inverse_mel_scale.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `inverse_mel_scale_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make inverse_mel_scale          # 编译 -> inverse_mel_scale/inverse_mel_scale.riscv
make inverse_mel_scale.spike    # 在 Spike 上运行
make gen-inverse_mel_scale      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 281 |
| max\|diff\| | 1e-06 |
| max\|ref\| | 16.4528 |

