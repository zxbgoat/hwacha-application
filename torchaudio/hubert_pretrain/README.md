# hubert_pretrain

torchaudio 的 `torchaudio.models.HuBERTPretrainModel` 在 Hwacha 上的一次前向，与 PyTorch 逐元素比对。文档：https://docs.pytorch.org/audio/stable/generated/torchaudio.models.HuBERTPretrainModel.html

用例：hubert_pretrain_model(...)：wav2vec2 特征提取 + 掩码生成器（mask_prob 0.5、mask_length 2）+ transformer + logit 生成器（10 类、final_dim 8），输入 400 个采样与 19 个标签，输出 [softmax(logit_m) | softmax(logit_u) | 特征惩罚]。

**注意**：MaskGenerator 在 forward 里用 torch 的随机数抽掩码，LogitGenerator 用布尔索引取被掩 / 未掩的行（形状数据相关），_compute_logits 用 -inf 填充。导出图：掩码固定为 torch.manual_seed(0) 下的那次抽样（参考也在同一种子下调用），掩码嵌入用 torch.where 写入，logits 对全部帧计算后用常量 one-hot 矩阵选行，-inf 换成 -1e4（softmax 相同，避免选行矩阵乘里的 0·(-inf)）；输出取 softmax 以避免比较 -inf。参考值由真正的 torchaudio 前向算出，导出前脚本断言两者一致。

来源：`export_ta.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机权重与输入、固定种子；页面上的工厂函数构建论文尺寸的模型，远超 Spike 的规模，这里用微型尺寸实例化同一架构。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x400 |
| 输出 | 210 |

## 文件

| 文件 | 内容 |
|---|---|
| `hubert_pretrain.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `hubert_pretrain_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `hwlib.s` | 库内核 |
| `README.md` | 本文件 |

## 编译与运行

```
make hubert_pretrain          # 编译 -> hubert_pretrain/hubert_pretrain.riscv
make hubert_pretrain.spike    # 在 Spike 上运行
make gen-hubert_pretrain      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 33,568 |
| max\|diff\| | 0 |
| max\|ref\| | 0.819255 |

