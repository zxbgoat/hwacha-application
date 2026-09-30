# yolov5n_p6

Ultralytics YOLOv5 `yolov5n-p6.yaml` `n` 规模（nano，目标检测（P3-P6 四级输出，多一级 stride 64），检测头 `Detect`）在 Hwacha 上的一次前向，与 PyTorch 比对。模型：https://github.com/ultralytics/yolov5 ；任务文档：https://docs.ultralytics.com/tasks/detect

比对内容：one2one 检测头在全部 anchor（P3-P6 四级）上的解码输出（1 x 84 x anchors），逐元素比对；top-k 选择留给 host。

权重随机（固定种子；BatchNorm 给随机的 running 统计量、卷积偏置随机抽取，使前向非退化），输入随机；同一组权重同时用于 PyTorch 参考与 Hwacha 构建。

## 本 case 的特殊处理

- YOLOv5 不是 end2end（没有 one2one 分支），推理走 one2many 头。 导出图以 `export=True` 走 Detect 的导出路径，并把 `postprocess`（按分数 top-k 选 max_det 个 anchor 再 gather）替换为恒等，网络返回全部 anchor；选择留给 host（与 ../torchvision 的检测 case 一致）。
- `reg_max=16`：回归分支输出 16 个 bin 的 logits，DFL 对每个 bin 做 softmax 后求期望得到距离，`dist2bbox` 在 anchor 网格上解码成 xywh 像素坐标。
- 本 case 带 `HWMLIRFLAGS`（`--unroll-small=2`）：v5 的 stem 是 6x6 stride-2 卷积（其余家族是 3x3），逐 tap 完全展开时 hwacha-cc 报 `out of Hwacha registers of class vs (nothing to spill) in net_kernel_2`，`--collapse-all` 也救不回来，`gen.sh` 的重试链落到 `--unroll-small=2`（只展开 2 倍）。
- anchor 网格与 stride 是 `make_anchors` 生成的常量。torch-mlir 的 fx importer 把常量 tensor 经 `tensor.tolist()` 转成 literal，导出脚本改为直接经 numpy 取原始缓冲区（`_tensor_to_numpy`）；Detect 缓存的 anchor 表是 `make_anchors(...).transpose(0, 1)`，非连续，必须先 `ascontiguousarray`，否则常量按转置后的内存布局落盘，每个 anchor 的 x / y 坐标互换（yolo26 首版 max|diff| = 128 即由此而来）。
- 输入取 128x128：P6 级 stride 64，64x64 输入下 P6 只剩 1x1。

## 形状与规模

| | 值 |
|---|---|
| 输入 | 1x3x128x128（49,152 个 float）|
| 输出元素数 | 28,560 |
| 参数量 | 4.3M |
| 权重 blob | 17 MB |
| 缩放常数 [depth, width, max_channels] | `[0.50, 0.25, 1024]` |

## 文件

| 文件 | 内容 |
|---|---|
| `yolov5n_p6.s` | hwacha-cc 生成的汇编，入口 `net` |
| `yolov5n_p6_weights.bin.S` | 权重的 `.incbin` 桩，按符号切分 blob；前半段放 `.weights_lo`、后半段放 `.weights_hi`（`split_weights.py`） |
| `yolov5n_p6_weights.bin` | 权重 blob（不入 git，`make gen-yolov5n_p6` 按固定种子逐字节重建） |
| `yolov5n_p6_check.bin` | 输入与 PyTorch 参考输出，host 用 `.incbin` 内嵌 |
| `yolo_main.c` | 通用 host：调用 `net`，比对 max\|diff\|（容差 1e-4 + 1e-2·max\|ref\|）与 argmax（分类输出） |
| `hwlib.s` | 卷积 / 池化库内核 |
| `HWMLIRFLAGS` | 本 case 需要的 hwacha-mlir 映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make yolov5n_p6          # 编译 -> yolov5n_p6/yolov5n_p6.riscv
make yolov5n_p6.spike    # 在 Spike 上运行（内存按权重大小自动确定）
make gen-yolov5n_p6      # 从 PyTorch 重新生成汇编、权重与参考
```

hwacha-mlir 映射：`--unroll-small=2`。

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 9,435,643 |
| max\|diff\| | 6.1e-05 |
| max\|ref\| | 1084.05 |
| argmax（硬件 / 参考） | 1019 / 1019 |

