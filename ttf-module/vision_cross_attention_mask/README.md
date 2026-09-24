# vision_cross_attention_mask

torchtune 的 `torchtune.modules.transforms.VisionCrossAttentionMask` 在 Hwacha 上的一次前向，与 PyTorch 逐元素比对。文档：https://meta-pytorch.org/torchtune/0.6/generated/torchtune.modules.transforms.VisionCrossAttentionMask.html

用例：由 token 列表 [image_token, 7 个文本 token] 与 1 张 8x8 图（patch 4：4 个 patch + CLS）生成文本 -> 图像掩码，作为交叉注意力层的 encoder_mask。

来源：`export_ttf.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机权重与输入、固定种子；规模很小（embed 32、4 头 x head_dim 8、序列 8、词表 16）。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x8x32 |
| 输出 | 1x8x32 |

## 文件

| 文件 | 内容 |
|---|---|
| `vision_cross_attention_mask.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `vision_cross_attention_mask_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make vision_cross_attention_mask          # 编译 -> vision_cross_attention_mask/vision_cross_attention_mask.riscv
make vision_cross_attention_mask.spike    # 在 Spike 上运行
make gen-vision_cross_attention_mask      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 7,792 |
| max\|diff\| | 0 |
| max\|ref\| | 3.39269 |

