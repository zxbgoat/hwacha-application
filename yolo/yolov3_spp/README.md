# yolov3_spp

Ultralytics YOLOv3 `yolov3-spp.yaml` （目标检测（SPP 变体：P5 输出前插入 SPP 空间金字塔池化，yolov3.yaml 本身没有），检测头 `Detect`）在 Hwacha 上的一次前向，与 PyTorch 比对。模型：https://github.com/ultralytics/yolov3 ；任务文档：https://docs.ultralytics.com/tasks/detect

比对内容：检测头在全部 anchor 上的解码输出（1 x 84 x anchors，P3-P5 三级），逐元素比对；top-k 选择留给 host。

权重随机（固定种子；BatchNorm 给随机的 running 统计量、卷积偏置随机抽取，使前向非退化），输入随机；同一组权重同时用于 PyTorch 参考与 Hwacha 构建。

## 本 case 的特殊处理

- YOLOv3 不是 end2end（没有 one2one 分支），推理走 one2many 头。 导出图以 `export=True` 走 Detect 的导出路径，并把 `postprocess`（按分数 top-k 选 max_det 个 anchor 再 gather）替换为恒等，网络返回全部 anchor；选择留给 host（与 ../torchvision 的检测 case 一致）。
- `reg_max=16`：回归分支输出 16 个 bin 的 logits，DFL 对每个 bin 做 softmax 后求期望得到距离，`dist2bbox` 在 anchor 网格上解码成 xywh 像素坐标。
- anchor 网格与 stride 是 `make_anchors` 生成的常量。torch-mlir 的 fx importer 把常量 tensor 经 `tensor.tolist()` 转成 literal，导出脚本改为直接经 numpy 取原始缓冲区（`_tensor_to_numpy`）；Detect 缓存的 anchor 表是 `make_anchors(...).transpose(0, 1)`，非连续，必须先 `ascontiguousarray`，否则常量按转置后的内存布局落盘，每个 anchor 的 x / y 坐标互换（yolo26 首版 max|diff| = 128 即由此而来）。

## 形状与规模

| | 值 |
|---|---|
| 输入 | 1x3x64x64（12,288 个 float）|
| 输出元素数 | 7,056 |
| 参数量 | 104.8M |
| 权重 blob | 400 MB |
| depth_multiple / width_multiple | `1.0 / 1.0`（v3 用这两个常量，没有 compound scaling）|

## 文件

| 文件 | 内容 |
|---|---|
| `yolov3_spp.s` | hwacha-cc 生成的汇编，入口 `net` |
| `yolov3_spp_weights.bin.S` | 权重的 `.incbin` 桩，按符号切分 blob；前半段放 `.weights_lo`、后半段放 `.weights_hi`（`split_weights.py`） |
| `yolov3_spp_weights.bin` | 权重 blob（不入 git，`make gen-yolov3_spp` 按固定种子逐字节重建） |
| `yolov3_spp_check.bin` | 输入与 PyTorch 参考输出，host 用 `.incbin` 内嵌 |
| `yolo_main.c` | 通用 host：调用 `net`，比对 max\|diff\|（容差 1e-4 + 1e-2·max\|ref\|）与 argmax（分类输出） |
| `hwlib.s` | 卷积 / 池化库内核 |
| `README.md` | 本文件 |

## 编译与运行

```
make yolov3_spp          # 编译 -> yolov3_spp/yolov3_spp.riscv
make yolov3_spp.spike    # 在 Spike 上运行（内存按权重大小自动确定）
make gen-yolov3_spp      # 从 PyTorch 重新生成汇编、权重与参考
```

hwacha-mlir 映射：`默认（最内维为 lane）`。

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 16,479,161 |
| max\|diff\| | 3.1e-05 |
| max\|ref\| | 563.786 |
| argmax（硬件 / 参考） | 333 / 333 |

