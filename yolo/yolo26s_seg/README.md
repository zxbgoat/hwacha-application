# yolo26s_seg

Ultralytics YOLO26 `yolo26-seg.yaml` 的 `s` 规模（small，实例分割，检测头 `Segment26`）在 Hwacha 上的一次前向，与 PyTorch 比对。模型：https://github.com/ultralytics/yolo26 ；任务文档：https://docs.ultralytics.com/tasks/segment

比对内容：检测头输出（1 x (4 + 80 + 32 个 mask 系数) x anchors）与 Proto26 的原型 mask（1 x 32 x H/4 x W/4）拼成一行，逐元素比对。

权重随机（固定种子；BatchNorm 给随机的 running 统计量、卷积偏置随机抽取，使前向非退化），输入随机；同一组权重同时用于 PyTorch 参考与 Hwacha 构建。

## 本 case 的特殊处理

- YOLO26 为 NMS-free（end2end）：推理用 one2one 头，导出图以 `export=True` 走 Detect 的导出路径并把 `postprocess`（按分数 top-k 选 max_det 个 anchor 再 gather）替换为恒等，网络返回全部 anchor；选择留给 host（与 ../torchvision 的检测 case 一致）。
- `reg_max=1`，DFL 为恒等：回归分支直接输出 4 个距离，`dist2bbox` 在 anchor 网格上解码成 xyxy 像素坐标；anchor 网格与 stride 是 `make_anchors` 生成的常量。torch-mlir 的 fx importer 把常量 tensor 经 `tensor.tolist()` 转成 literal，导出脚本改为直接经 numpy 取原始缓冲区（`_tensor_to_numpy`）；Detect 缓存的 anchor 表是 `make_anchors(...).transpose(0, 1)`，非连续，必须先 `ascontiguousarray`，否则常量按转置后的内存布局落盘，每个 anchor 的 x / y 坐标互换（首版 max|diff| = 128 即由此而来）。
- Segment26 返回 `(预测, 原型 mask)` 两个输出，Wrap 把它们各自展平后拼成一行。

## 形状与规模

| | 值 |
|---|---|
| 输入 | 1x3x64x64（12,288 个 float）|
| 输出元素数 | 17,936 |
| 参数量 | 11.5M |
| 权重 blob | 40 MB |
| 缩放常数 [depth, width, max_channels] | `[0.50, 0.50, 1024]` |

## 文件

| 文件 | 内容 |
|---|---|
| `yolo26s_seg.s` | hwacha-cc 生成的汇编，入口 `net` |
| `yolo26s_seg_weights.bin.S` | 权重的 `.incbin` 桩，按符号切分 blob；前半段放 `.weights_lo`、后半段放 `.weights_hi`（`split_weights.py`） |
| `yolo26s_seg_weights.bin` | 权重 blob（不入 git，`make gen-yolo26s_seg` 按固定种子逐字节重建） |
| `yolo26s_seg_check.bin` | 输入与 PyTorch 参考输出，host 用 `.incbin` 内嵌 |
| `yolo_main.c` | 通用 host：调用 `net`，比对 max\|diff\|（容差 1e-4 + 1e-2·max\|ref\|）与 argmax（分类输出） |
| `hwlib.s` | 卷积 / 池化库内核 |
| `README.md` | 本文件 |

## 编译与运行

```
make yolo26s_seg          # 编译 -> yolo26s_seg/yolo26s_seg.riscv
make yolo26s_seg.spike    # 在 Spike 上运行（内存按权重大小自动确定）
make gen-yolo26s_seg      # 从 PyTorch 重新生成汇编、权重与参考
```

hwacha-mlir 映射：`默认（最内维为 lane）`。

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 32,525,608 |
| max\|diff\| | 2e-06 |
| max\|ref\| | 92.6256 |
| argmax（硬件 / 参考） | 9630 / 9630 |

