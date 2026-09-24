# clip_vision_encoder

torchtune 模型参考页的 `torchtune.models.clip.clip_vision_encoder` 所用架构在 Hwacha 上的一次前向，与 PyTorch 逐元素比对。文档：https://meta-pytorch.org/torchtune/0.6/generated/torchtune.models.clip.clip_vision_encoder.html

用例：CLIP 视觉编码器：tile 8、patch 4（4 个 patch + CLS）、2 层、embed 32，输出 token 序列。

**注意**：单 tile（max_num_tiles=1）：平铺位置编码按 aspect ratio 循环，torch.export 无法追踪。参考值由真正的 torchtune 前向算出，导出前脚本断言两者一致。

来源：`export_ttm.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机权重与输入、固定种子。带尺寸的构建函数（7b、0.5b …）是同一组件构建函数加固定超参数，远超 Spike 的规模，这里用微型尺寸实例化同一架构。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x1x1x3x8x8 |
| 输出 | 1x1x1x5x32 |

## 文件

| 文件 | 内容 |
|---|---|
| `clip_vision_encoder.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `clip_vision_encoder_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `hwlib.s` | 库内核 |
| `README.md` | 本文件 |

## 编译与运行

```
make clip_vision_encoder          # 编译 -> clip_vision_encoder/clip_vision_encoder.riscv
make clip_vision_encoder.spike    # 在 Spike 上运行
make gen-clip_vision_encoder      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 21,875 |
| max\|diff\| | 0 |
| max\|ref\| | 2.42655 |

