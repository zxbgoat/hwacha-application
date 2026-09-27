# mvit_v1_b

torchvision `mvit_v1_b`（视频分类）在 Hwacha 上的一次前向，与 PyTorch 比对。文档：https://docs.pytorch.org/vision/stable/models.html#video-classification

比对内容：400 类 logits，逐元素比对并要求 argmax 一致。

权重随机（固定种子，BatchNorm 给随机的 running 统计量使前向非退化），输入随机；同一组权重同时用于 PyTorch 参考与 Hwacha 构建。

## 本 case 的特殊处理

- MViT 的 pooling attention 用逐通道 Conv3d（groups = 通道数），5-D 输入的分组 aten.convolution 被 torch-mlir 标为 illegal；导出图把它按时间核的每个 tap 拆成对 B·T 帧的 2-D 逐通道卷积再求和（与 nn.Conv3d 一致到 1e-7）。构造函数硬编码 `spatial_size=(224, 224)`（位置编码按 patch 网格定尺寸），导出时给 `_mvit` 打补丁按 32x32 clip 构建同一块配置。

## 形状与规模

| | 值 |
|---|---|
| 输入 | 1x3x16x32x32（49,152 个 float）|
| 输出元素数 | 400 |
| 参数量 | 36.6M |
| 权重 blob | 139 MB |
| 帧数 | 16 |

## 文件

| 文件 | 内容 |
|---|---|
| `mvit_v1_b_tv.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mvit_v1_b_tv_weights.bin.S` | 权重的 `.incbin` 桩，按符号切分 blob；前半段放 `.weights_lo`、后半段放 `.weights_hi`（`split_weights.py`） |
| `mvit_v1_b_tv_weights.bin` | 权重 blob（不入 git，`make gen-mvit_v1_b` 按固定种子逐字节重建） |
| `mvit_v1_b_tv_check.bin` | 输入与 PyTorch 参考输出，host 用 `.incbin` 内嵌 |
| `tv_main.c` | 通用 host：调用 `net`，比对 max\|diff\|（容差 1e-4 + 1e-2·max\|ref\|）与 argmax（分类输出） |
| `hwlib.s` | 卷积 / 池化库内核 |
| `HWMLIRFLAGS` | 本 case 需要的 hwacha-mlir 映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make mvit_v1_b          # 编译 -> mvit_v1_b/mvit_v1_b_tv.riscv
make mvit_v1_b.spike    # 在 Spike 上运行（内存按权重大小自动确定）
make gen-mvit_v1_b      # 从 PyTorch 重新生成汇编、权重与参考
```

hwacha-mlir 映射：`--unroll-small=2`。

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 88,803,152 |
| max\|diff\| | 2e-06 |
| max\|ref\| | 1.98928 |
| argmax（硬件 / 参考） | 236 / 236 |

