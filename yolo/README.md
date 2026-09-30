# Ultralytics YOLOv5 / YOLOv8 / YOLO11 / YOLO26 on Hwacha

The models of github.com/ultralytics/yolov5 (the `ultralytics` package, `cfg/models/v5/*.yaml`),
github.com/ultralytics/yolov8 (`cfg/models/v8/*.yaml`), github.com/ultralytics/yolo11 (`cfg/models/11/*.yaml`)
and github.com/ultralytics/yolo26 (`cfg/models/26/*.yaml`), built from their yaml with random weights, run
through PyTorch -> torch-mlir (linalg on tensors) -> hwacha-mlir -> hwacha-cc and checked on Spike against
PyTorch's own forward (fixed seed). One directory per case, `yolov5|yolov8|yolo11|yolo26<scale>[_<task>]`, the
5 scales n / s / m / l / x of:

| family | tasks | cases |
|---|---|---|
| `yolov5` | detection and its P6 variant (`_p6`); the package's v5 yamls are detection only | 10 |
| `yolov8` | detection, instance segmentation (`-seg`), classification (`-cls`), pose (`-pose`), oriented detection (`-obb`), the P2 / P6 detection variants and the P6 variants of seg / pose (`_seg_p6`, `_pose_p6`) | 45 |
| `yolo11` | detection, `-seg`, `-cls`, `-pose`, `-obb` | 25 |
| `yolo26` | the same five, plus semantic segmentation (`-sem`), depth (`-depth`) and the P2 / P6 detection variants | 45 |

YOLO11 has no semantic-segmentation, depth or P2 / P6 yaml; YOLOv8 has neither sem nor depth but does have
seg-p6 / pose-p6; YOLOv5 ships only `yolov5.yaml` and `yolov5-p6.yaml` (no seg / cls / pose / obb --
`yolov5nu.yaml` is just the `yolov5n.yaml` alias that `check_yaml` resolves through the unified backbone
name). Only YOLO26 is end2end. Not covered: the v8 yamls that swap in a foreign backbone or need a text
encoder (`-cls-resnet50/101`, `-ghost*`, `-rtdetr`, `-world*`, `yoloe-*`). Each `<case>/` is self-contained:

| file | what |
|---|---|
| `<case>.s` | hwacha-cc assembly of the network (`net`) |
| `<case>_weights.bin` + `.bin.S` | the parameters, stripped to a blob and pulled in with `.incbin` |
| `<case>_check.bin` | the input and PyTorch's output, embedded by the host with `.incbin` |
| `yolo_main.c` | generic host: runs `net`, compares max\|diff\| against 1e-4 + 1e-2·max\|ref\| (and the argmax for the classification logits); prints the first mismatches on failure |
| `hwlib.s` | conv / pool / norm library kernels every model links |
| `HWMLIRFLAGS` | present when the case needed `--collapse-all` / `--unroll-small=2` (hwacha-cc ran out of registers otherwise) |

The weight blobs (`*_weights.bin`) are not in git; after a fresh clone run `make gen-<case>` (or `make gen-all`)
to recreate them -- the export uses a fixed seed, so the regenerated assembly and weights are byte-identical
to the committed ones.

`make` builds all, `make run` runs all on Spike, `make <case>` / `make <case>.spike` for one. `MEM=<MB>`
overrides the simulator memory (by default sized per case from its weights).

`make gen-<case>` regenerates a case from PyTorch (`gen.sh`: export_yolo.py -> mlir-opt bufferize ->
hwacha-mlir --weights-bin -> hwacha-cc, retrying with `--collapse-all` and `--unroll-small=2` when hwacha-cc
runs out of registers); `models.txt` lists each case's input size, `make gen-all` does every case that has no
assembly yet. It needs the hwacha-cc build tree, mlir-opt and the torch-mlir venv (`../.tmenv`, with
`ultralytics` 8.4.165 installed in it).

## What is compared

Inputs are 1x3x64x64 (P5 = 2x2; 128x128 for the P6 models, stride 64). The exported network is the eval
forward with the head in export mode, i.e. what the post-processing consumes, over **every anchor**; every
output tensor is flattened and concatenated. The families differ in the detection head: YOLO26
(`end2end: True`, `reg_max: 1`) is NMS-free, so the inference head is the one2one branch and there is no DFL;
YOLOv5, YOLOv8 and YOLO11 have no one2one branch (`end2end: False`) and infer from one2many, with
`reg_max: 16` running the DFL softmax over the 16 bins. Boxes are xyxy in pixels for yolo26, xywh for
yolov5 / yolov8 / yolo11. Their `Detect` head is one class; the families differ in the backbone / neck blocks
(v5: C3 + a 6x6 stem convolution; v8: C2f; 11: C3k2 + C2PSA; 26: C3k2 + C2PSA with the end2end head and the
extra task heads).

| task | head | output (64x64 input, 84 anchors = 8² + 4² + 2²) |
|---|---|---|
| detection, P2, P6 (v5: detect, p6) | `Detect` | (1, anchors, 4 + 80): pixel boxes (`dist2bbox` of the distances with the anchor grid × stride), sigmoid class scores |
| instance segmentation (+ P6 for v8) | `Segment26` / `Segment` | (1, 84, 4 + 80 + 32 mask coefficients) + the prototype masks (1, 32, 16, 16) |
| pose (+ P6 for v8) | `Pose26` / `Pose` | (1, 84, 4 + nc + 17·3): decoded keypoints, sigmoid visibility; `nc` = 80 (yolo26 / yolo11) or 1 (`yolov8-pose.yaml`: person only), so v8 pose has 56 channels |
| oriented detection | `OBB26` / `OBB` | (1, 84, 4 + 80 + 1): the box distances and the raw angle channel |
| semantic segmentation (yolo26 only) | `SemanticSegment` | the class logits (1, 19, 8, 8) at stride 8 |
| depth (yolo26 only) | `Depth` | the calibrated depth map (1, 1, 16, 16) at stride 4, `exp` of the clamped head output |
| classification | `Classify` | the logits (1, 1000); the host also compares the argmax |

`Detect.postprocess` (top-`max_det` over the class scores of every anchor, a `topk` + `gather` whose result
order is data-dependent) is replaced by the identity: the selection is left to the host, as in the torchvision
detection cases.

## Export notes

- Untrained BatchNorm sits at its zero-mean / unit-variance fixed point (outputs ~0), so BN gets random running
  stats and affine parameters as in `../torchvision`. The heads' final 1x1 convs are the only convs with a bias
  and `Detect.bias_init` sets the class bias to `log(5/nc/(640/stride)²)` ≈ −11.5 (sigmoid scores ~1e-5, below
  anything the tolerance could see), so every conv bias is drawn at random instead. The same weights feed the
  PyTorch reference and the Hwacha build.
- torch-mlir's fx importer turns every constant tensor into a literal via `np.array(tensor.tolist())` (one
  Python object per element); `export_yolo.py` converts through numpy directly, as `../torchvision/export_tv.py`
  does. The importer then hands the array's raw buffer to `DenseResourceElementsAttr`, so the array has to be
  C-contiguous: `Detect` caches its anchor grid as `make_anchors(...).transpose(0, 1)`, a strided view, and
  without `np.ascontiguousarray` the anchors landed transposed in memory -- the backbone and the class scores
  matched to the bit while every box coordinate was off by a multiple of the stride. (The torchvision / video /
  deformable exports only ever lift contiguous parameters, which is why the same patch never bit there.)
- `ultralytics` is imported with `YOLO_OFFLINE=1`: nothing is downloaded, the models are built from the yaml.
- No hwacha-mlir / hwacha-cc changes were needed for any family: the C3 (v5) / C2f (v8) / C3k2 + C2PSA
  (11, 26) blocks, the 6x6 stride-2 stem convolution of v5, the SPPF, the attention blocks, the
  nearest-neighbour upsampling, the bilinear `align_corners=True` upsampling of the depth head, the DFL
  softmax-expectation of yolov5 / yolov8 / yolo11 and the head decoding all lower with the existing kernels.
- The non-26 families use the package's head classes (`Detect` / `Segment` / `Pose` / `OBB`; yolo26 uses
  `Segment26` / `Pose26` / `OBB26`); `export_yolo.py` picks the model class by task and all families go
  through the same `Detect.postprocess` = identity patch. Where their decoding differs it is carried by both
  the reference and the generated code: keypoints are `(2k + anchor − 0.5)·stride` for yolov8 / yolo11
  (`Pose.kpts_decode`) and `(k + anchor)·stride` for yolo26 (`Pose26`), and boxes xywh vs xyxy.
- Case names use `_` where the yaml uses `-` (`yolov8n_seg_p6` <-> `yolov8n-seg-p6.yaml`); `export_yolo.py`
  maps the task suffix back to the yaml name.
- The 10 YOLOv5 cases are the only ones in the suite with an `HWMLIRFLAGS`: hwacha-cc reports
  `out of Hwacha registers of class vs (nothing to spill) in net_kernel_2`. That kernel is the 6x6 stride-2
  stem convolution, which v5 alone has; its fully unrolled taps need more vs registers than exist, and
  there is nothing left to spill. `--collapse-all` does not help, so `gen.sh`'s retry chain lands on
  `--unroll-small=2` (unroll the kernel body only 2x). The other families' first convolution is 3x3 and fits.

## Coverage (2026-09-30)

All 125 cases PASS on Spike (yolo26 max|diff| ≤ 1.7e-5, yolov5 / yolov8 / yolo11 ≤ 1.2e-4; the DFL distances
of v5 / v8 / 11 are larger in absolute terms, so their tolerance 1e-4 + 1e-2·max|ref| is looser); argmax
matches where checked. Only the 10 yolov5 cases needed `HWMLIRFLAGS` (`--unroll-small=2`, see the export
notes). Spike cycles (`rdcycle` around `net`) and weight blob size per scale:

### YOLOv5 (10 cases)

| task | outputs | n | s | m | l | x |
|---|---|---|---|---|---|---|
| detection | 7,056 | 5.1M / 10 MB | 14.9M / 34 MB | 32.9M / 95 MB | 59.8M / 203 MB | 96.9M / 371 MB |
| detection P6 (128x128) | 28,560 | 9.4M / 16 MB | 26.3M / 58 MB | 57.0M / 157 MB | 102.4M / 328 MB | 164.4M / 593 MB |

### YOLOv8 (45 cases)

| task | outputs | n | s | m | l | x |
|---|---|---|---|---|---|---|
| detection | 7,056 | 4.1M / 12 MB | 13.4M / 42 MB | 23.7M / 98 MB | 32.9M / 166 MB | 49.4M / 260 MB |
| instance segmentation | 17,936 | 5.8M / 13 MB | 19.6M / 45 MB | 37.3M / 104 MB | 56.8M / 175 MB | 86.5M / 274 MB |
| classification | 1,000 | 2.8M / 10 MB | 7.2M / 24 MB | 15.7M / 65 MB | 29.1M / 143 MB | 42.7M / 219 MB |
| pose | 4,704 | 4.1M / 12 MB | 13.4M / 44 MB | 23.6M / 101 MB | 32.8M / 169 MB | 49.3M / 265 MB |
| oriented detection | 7,140 | 4.1M / 12 MB | 13.5M / 43 MB | 23.8M / 101 MB | 33.0M / 170 MB | 49.5M / 265 MB |
| detection P2 | 28,560 | 4.8M / 12 MB | 14.4M / 41 MB | 25.6M / 95 MB | 36.2M / 163 MB | 53.9M / 254 MB |
| detection P6 (128x128) | 28,560 | 6.9M / 19 MB | 21.1M / 68 MB | 39.8M / 171 MB | 49.2M / 238 MB | 72.3M / 371 MB |
| instance segmentation P6 (128x128) | 72,208 | 10.3M / 20 MB | 33.3M / 71 MB | 66.5M / 177 MB | 98.1M / 248 MB | 144.9M / 387 MB |
| pose P6 (128x128) | 19,040 | 7.0M / 19 MB | 21.2M / 70 MB | 39.7M / 174 MB | 49.1M / 242 MB | 72.2M / 378 MB |

### YOLO11 (25 cases)

| task | outputs | n | s | m | l | x |
|---|---|---|---|---|---|---|
| detection | 7,056 | 6.9M / 10 MB | 23.7M / 36 MB | 46.8M / 76 MB | 56.1M / 97 MB | 120.6M / 217 MB |
| instance segmentation | 17,936 | 8.6M / 11 MB | 29.9M / 38 MB | 70.7M / 85 MB | 80.0M / 105 MB | 173.8M / 237 MB |
| classification | 1,000 | 4.9M / 10 MB | 15.2M / 25 MB | 23.1M / 44 MB | 28.4M / 53 MB | 59.0M / 113 MB |
| pose | 11,340 | 7.0M / 11 MB | 23.9M / 38 MB | 47.0M / 80 MB | 56.2M / 100 MB | 120.8M / 225 MB |
| oriented detection | 7,140 | 6.9M / 10 MB | 23.8M / 37 MB | 46.9M / 80 MB | 56.1M / 100 MB | 120.7M / 224 MB |

### YOLO26 (45 cases)

| task | outputs | n | s | m | l | x |
|---|---|---|---|---|---|---|
| detection | 7,056 | 7.4M / 9 MB | 25.9M / 36 MB | 47.2M / 78 MB | 55.3M / 94 MB | 118.9M / 213 MB |
| instance segmentation | 17,936 | 9.3M / 10 MB | 32.5M / 39 MB | 72.3M / 90 MB | 80.4M / 107 MB | 174.7M / 240 MB |
| semantic segmentation | 1,216 | 5.1M / 5 MB | 18.0M / 23 MB | 32.7M / 50 MB | 39.6M / 63 MB | 85.1M / 143 MB |
| depth | 256 | 14.0M / 19 MB | 32.3M / 46 MB | 52.2M / 84 MB | 60.3M / 101 MB | 121.6M / 213 MB |
| classification | 1,000 | 4.9M / 10 MB | 15.2M / 25 MB | 23.1M / 44 MB | 28.4M / 53 MB | 59.0M / 113 MB |
| pose | 11,340 | 7.6M / 11 MB | 26.0M / 39 MB | 47.3M / 82 MB | 55.5M / 99 MB | 119.1M / 220 MB |
| oriented detection | 7,140 | 7.4M / 9 MB | 25.9M / 37 MB | 47.3M / 81 MB | 55.4M / 98 MB | 119.0M / 220 MB |
| detection P2 | 28,560 | 8.3M / 9 MB | 27.2M / 36 MB | 50.9M / 77 MB | 59.9M / 95 MB | 127.8M / 215 MB |
| detection P6 (128x128) | 28,560 | 12.6M / 14 MB | 42.6M / 58 MB | 86.2M / 116 MB | 100.6M / 142 MB | 214.7M / 320 MB |

The weights total 12 GB (yolov8 6.0 GB, the largest blobs its x-scale P6 cases at 372-388 MB); the largest
cases run in a few minutes on Spike. Per-case
result lines are in `.logs/run_full.txt` (`../tools/run_full.sh yolo` regenerates it) and each
`<case>/README.md` (`../tools/gen_case_readme.py yolo`).
