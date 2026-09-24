# lora_llama3_2_vision_decoder

torchtune 模型参考页的 `torchtune.models.llama3_2_vision.lora_llama3_2_vision_decoder` 所用架构在 Hwacha 上的一次前向，与 PyTorch 逐元素比对。文档：https://meta-pytorch.org/torchtune/0.6/generated/torchtune.models.llama3_2_vision.lora_llama3_2_vision_decoder.html

用例：LoRA Llama 3.2 Vision 融合解码器（decoder_lora、fusion_lora），LoRA 加在 q_proj / v_proj 和 MLP 上（rank 4、alpha 8），lora_b 随机初始化（torchtune 默认置零，adapter 是恒等）。

**注意**：同 llama3_2_vision_decoder：FusionEmbedding 换成 one-hot 等价模块。参考值由真正的 torchtune 前向算出，导出前脚本断言两者一致。

来源：`export_ttm.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机权重与输入、固定种子。带尺寸的构建函数（7b、0.5b …）是同一组件构建函数加固定超参数，远超 Spike 的规模，这里用微型尺寸实例化同一架构。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x8 |
| 输出 | 1x8x16 |

## 文件

| 文件 | 内容 |
|---|---|
| `lora_llama3_2_vision_decoder.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `lora_llama3_2_vision_decoder_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make lora_llama3_2_vision_decoder          # 编译 -> lora_llama3_2_vision_decoder/lora_llama3_2_vision_decoder.riscv
make lora_llama3_2_vision_decoder.spike    # 在 Spike 上运行
make gen-lora_llama3_2_vision_decoder      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 38,188 |
| max\|diff\| | 0 |
| max\|ref\| | 2.05792 |

