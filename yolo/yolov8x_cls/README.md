# yolov8x_cls

Ultralytics YOLOv8 `yolov8x-cls.yaml` `x` 规模（xlarge，图像分类，检测头 `Classify`）在 Hwacha 上的一次前向，与 PyTorch 比对。模型：https://github.com/ultralytics/yolov8 ；任务文档：https://docs.ultralytics.com/tasks/classify

比对内容：1000 类 logits（softmax 之前），逐元素比对并要求 argmax 一致。

权重随机（固定种子；BatchNorm 给随机的 running 统计量、卷积偏置随机抽取，使前向非退化），输入随机；同一组权重同时用于 PyTorch 参考与 Hwacha 构建。

## 本 case 的特殊处理

- Classify 在 eval 下返回 `(softmax, logits)`，取 logits 比对（host 还比对 argmax）。

## 形状与规模

| | 值 |
|---|---|
| 输入 | 1x3x64x64（12,288 个 float）|
| 输出元素数 | 1,000 |
| 参数量 | 57.4M |
| 权重 blob | 219 MB |
| 缩放常数 [depth, width, max_channels] | `[1.00, 1.50, 512]` |

## 文件

| 文件 | 内容 |
|---|---|
| `yolov8x_cls.s` | hwacha-cc 生成的汇编，入口 `net` |
| `yolov8x_cls_weights.bin.S` | 权重的 `.incbin` 桩，按符号切分 blob；前半段放 `.weights_lo`、后半段放 `.weights_hi`（`split_weights.py`） |
| `yolov8x_cls_weights.bin` | 权重 blob（不入 git，`make gen-yolov8x_cls` 按固定种子逐字节重建） |
| `yolov8x_cls_check.bin` | 输入与 PyTorch 参考输出，host 用 `.incbin` 内嵌 |
| `yolo_main.c` | 通用 host：调用 `net`，比对 max\|diff\|（容差 1e-4 + 1e-2·max\|ref\|）与 argmax（分类输出） |
| `hwlib.s` | 卷积 / 池化库内核 |
| `README.md` | 本文件 |

## 编译与运行

```
make yolov8x_cls          # 编译 -> yolov8x_cls/yolov8x_cls.riscv
make yolov8x_cls.spike    # 在 Spike 上运行（内存按权重大小自动确定）
make gen-yolov8x_cls      # 从 PyTorch 重新生成汇编、权重与参考
```

hwacha-mlir 映射：`默认（最内维为 lane）`。

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 42,746,572 |
| max\|diff\| | 0 |
| max\|ref\| | 0.160376 |
| argmax（硬件 / 参考） | 479 / 479 |

