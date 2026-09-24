# register_fusion_module

torchtune 的 `torchtune.modules.model_fusion.register_fusion_module` 在 Hwacha 上的一次前向，与 PyTorch 逐元素比对。文档：https://meta-pytorch.org/torchtune/0.6/generated/torchtune.modules.model_fusion.register_fusion_module.html

用例：把交叉注意力层注册为融合模块（断言 fusion_params 为其全部参数），再前向。

来源：`export_ttf.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机权重与输入、固定种子；规模很小（embed 32、4 头 x head_dim 8、序列 8、词表 16）。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x8x32 |
| 输出 | 1x8x32 |

## 文件

| 文件 | 内容 |
|---|---|
| `register_fusion_module.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `register_fusion_module_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make register_fusion_module          # 编译 -> register_fusion_module/register_fusion_module.riscv
make register_fusion_module.spike    # 在 Spike 上运行
make gen-register_fusion_module      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 7,268 |
| max\|diff\| | 0 |
| max\|ref\| | 3.69312 |

