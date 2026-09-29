# yolo26l_depth

Ultralytics YOLO26 `yolo26-depth.yaml` 的 `l` 规模（large，深度估计，检测头 `Depth`）在 Hwacha 上的一次前向，与 PyTorch 比对。模型：https://github.com/ultralytics/yolo26 ；任务文档：https://docs.ultralytics.com/tasks/depth

比对内容：深度图（1 x 1 x H/4 x W/4，exp 后为正值），逐元素比对。

权重随机（固定种子；BatchNorm 给随机的 running 统计量、卷积偏置随机抽取，使前向非退化），输入随机；同一组权重同时用于 PyTorch 参考与 Hwacha 构建。

## 本 case 的特殊处理

- Depth 头输出 `exp(clamp(out, -4, 5))` 再乘以标定 buffer `cal_a` / `cal_b`（默认恒等）；不走 export 路径的 4 倍上采样。

## 形状与规模

| | 值 |
|---|---|
| 输入 | 1x3x64x64（12,288 个 float）|
| 输出元素数 | 256 |
| 参数量 | 26.5M |
| 权重 blob | 101 MB |
| 缩放常数 [depth, width, max_channels] | `[1.00, 1.00, 512]` |

## 文件

| 文件 | 内容 |
|---|---|
| `yolo26l_depth.s` | hwacha-cc 生成的汇编，入口 `net` |
| `yolo26l_depth_weights.bin.S` | 权重的 `.incbin` 桩，按符号切分 blob；前半段放 `.weights_lo`、后半段放 `.weights_hi`（`split_weights.py`） |
| `yolo26l_depth_weights.bin` | 权重 blob（不入 git，`make gen-yolo26l_depth` 按固定种子逐字节重建） |
| `yolo26l_depth_check.bin` | 输入与 PyTorch 参考输出，host 用 `.incbin` 内嵌 |
| `yolo_main.c` | 通用 host：调用 `net`，比对 max\|diff\|（容差 1e-4 + 1e-2·max\|ref\|）与 argmax（分类输出） |
| `hwlib.s` | 卷积 / 池化库内核 |
| `README.md` | 本文件 |

## 编译与运行

```
make yolo26l_depth          # 编译 -> yolo26l_depth/yolo26l_depth.riscv
make yolo26l_depth.spike    # 在 Spike 上运行（内存按权重大小自动确定）
make gen-yolo26l_depth      # 从 PyTorch 重新生成汇编、权重与参考
```

hwacha-mlir 映射：`默认（最内维为 lane）`。

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 60,276,019 |
| max\|diff\| | 0 |
| max\|ref\| | 0.296264 |
| argmax（硬件 / 参考） | 232 / 232 |

