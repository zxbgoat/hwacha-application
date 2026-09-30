# demo: ResNet-50 图像分类推理（Hwacha）

用 torchvision 的**预训练** ResNet-50（ImageNet-1K V2，acc@1 80.9%）对一张真实照片做分类推理，
整网编译到 Hwacha，在 Spike 上跑出 top-5 类别并与 PyTorch 逐元素比对 logits。

```
make gen        # 从 PyTorch 导出（预训练权重 + 图片预处理）—— 需要 torch-mlir venv
make            # 编译 -> resnet50.riscv
make run        # 在 Spike 上推理（约 2.5 分钟，93M 周期）
make gen IMG=path/to/other.jpg    # 换一张图重新生成
```

## 结果（dog.jpg）

```
ResNet-50 (ImageNet-1K V2 weights) on Hwacha, input 1x3x224x224
top-5:
   1. 44.0%  Samoyed (class 258)
   2.  2.2%  white wolf (class 270)
   3.  1.4%  Pomeranian (class 259)
   4.  0.8%  Great Pyrenees (class 257)
   5.  0.6%  Eskimo dog (class 248)
93301729 cycles; vs PyTorch: max|diff|=0.000005 max|ref|=6.800, argmax hw=258 ref=258 (Samoyed); ok
demo PASS
```

与 PyTorch 在同一张图上的输出一致：top-5 类别与概率完全相同（Samoyed 44.0%），
logits 最大绝对差 5e-6。这不是"随机权重跑通"的数值验证，而是真实模型、真实图片的分类结果。

## 文件

| 文件 | 内容 |
|---|---|
| `export.py` | 载入预训练 resnet50、按 `weights.transforms()` 预处理图片（resize 232 → center crop 224 → 归一化）、导出 linalg MLIR，并写出 `check.bin` 与 `labels.h` |
| `gen.sh` | export.py → mlir-opt bufferize → hwacha-mlir `--weights-bin` → hwacha-cc，产出 `resnet50.s` / 权重 / `check.bin` / `labels.h` |
| `main.c` | Spike 上的 host：读入图片张量、调用 `net`、softmax 后打印 top-5 类名，并与 PyTorch logits 比对 |
| `labels.h` | ImageNet-1K 的 1000 个类名（由 `weights.meta["categories"]` 生成） |
| `demo.ld` | 链接脚本（`../../torchvision/tv.ld` 改 `main.o`），权重分 `.weights_lo` / `.weights_hi` 两段 |
| `hwlib.s` | 卷积 / 池化库内核（与 torchvision 套件共用） |
| `dog.jpg` | 输入图片（PyTorch hub 的示例图） |

`resnet50.s`、`check.bin`、`labels.h`、`resnet50_weights.bin.S` 已入库，所以 `make && make run`
不需要 torch-mlir venv；`resnet50_weights.bin`（98 MB）按 `.gitignore` 不入库，用 `make gen` 重建。

## 与 `../../torchvision/` 的关系

torchvision 套件里的 `resnet50` case 用**随机权重 + 随机输入**验证"Hwacha 复现 PyTorch"；
这个 demo 换成预训练权重和真实图片，验证"真实任务在 Hwacha 上跑出正确分类"。两者的导出与编译
路径相同：`export_tv.py` / `export.py` → `mlir-opt` bufferize → `hwacha-mlir --weights-bin` →
`hwacha-cc`。

一个坑记录在这：Makefile 必须带

```
MAKEFLAGS += --no-builtin-rules
.SUFFIXES:
```

否则 make 的内建规则会把 `resnet50_weights.bin.S` 当成 `resnet50_weights.bin` 的源文件，用
编译链接出的 ELF 覆盖 98 MB 的权重 blob（表现为网络输出全是巨大值、NaN）。`tv.ld` 那套
Makefile 里已有这条设置，demo 起初漏了。
