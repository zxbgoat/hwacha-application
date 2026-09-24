# get_fusion_params

torchtune 的 `torchtune.modules.model_fusion.get_fusion_params` 在 Hwacha 上的一次前向，与 PyTorch 逐元素比对。文档：https://meta-pytorch.org/torchtune/0.6/generated/torchtune.modules.model_fusion.get_fusion_params.html

用例：对 DeepFusionModel 调用 get_fusion_params（断言只含 fusion_layer 的参数），再前向。

来源：`export_ttf.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机权重与输入、固定种子；规模很小（embed 32、4 头 x head_dim 8、序列 8、词表 16）。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x8 |
| 输出 | 1x8x16 |

## 文件

| 文件 | 内容 |
|---|---|
| `get_fusion_params.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `get_fusion_params_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make get_fusion_params          # 编译 -> get_fusion_params/get_fusion_params.riscv
make get_fusion_params.spike    # 在 Spike 上运行
make gen-get_fusion_params      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 17,294 |
| max\|diff\| | 0 |
| max\|ref\| | 1.27928 |

