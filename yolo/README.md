# Ultralytics YOLO11 / YOLO26 on Hwacha

The models of github.com/ultralytics/yolo11 (the `ultralytics` package, `cfg/models/11/*.yaml`) and
github.com/ultralytics/yolo26 (`cfg/models/26/*.yaml`), built from their yaml with random weights, run
through PyTorch -> torch-mlir (linalg on tensors) -> hwacha-mlir -> hwacha-cc and checked on Spike against
PyTorch's own forward (fixed seed). One directory per case, `yolo11|yolo26<scale>[_<task>]`, the 5 scales
n / s / m / l / x of:

| family | tasks | cases |
|---|---|---|
| `yolo11` | detection, instance segmentation (`-seg`), classification (`-cls`), pose (`-pose`), oriented detection (`-obb`) | 25 |
| `yolo26` | the same five, plus semantic segmentation (`-sem`), depth (`-depth`) and the P2 / P6 detection variants | 45 |

YOLO11 has no semantic-segmentation, depth or P2 / P6 yaml, and is not end2end. Each `<case>/` is
self-contained:

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
output tensor is flattened and concatenated. The two families differ in the detection head: YOLO26
(`end2end: True`, `reg_max: 1`) is NMS-free, so the inference head is the one2one branch and there is no DFL;
YOLO11 has no one2one branch (`end2end: False`) and infers from one2many, with `reg_max: 16` running the DFL
softmax over the 16 bins. Boxes are xyxy in pixels for yolo26, xywh for yolo11.

| task | head (yolo26 / yolo11) | output (64x64 input, 84 anchors = 8² + 4² + 2²) |
|---|---|---|
| detection, P2, P6 | `Detect` | (1, anchors, 4 + 80): pixel boxes (`dist2bbox` of the distances with the anchor grid × stride), sigmoid class scores |
| instance segmentation | `Segment26` / `Segment` | (1, 84, 4 + 80 + 32 mask coefficients) + the prototype masks (1, 32, 16, 16) |
| pose | `Pose26` / `Pose` | (1, 84, 4 + 80 + 17·3): decoded keypoints `(k·2 + anchor − 0.5) × stride`, sigmoid visibility |
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
- No hwacha-mlir / hwacha-cc changes were needed for either family: the C3k2 / C2PSA / SPPF / attention
  blocks, the nearest-neighbour upsampling, the bilinear `align_corners=True` upsampling of the depth head,
  the DFL argmax weights of yolo11 and the head decoding all lower with the existing kernels.
- YOLO11's head classes are the package's `Segment` / `Pose` / `OBB` (yolo26 uses `Segment26` / `Pose26` /
  `OBB26`); `export_yolo.py` picks the model class by task and both families go through the same
  `Detect.postprocess` = identity patch. Their decoding differs and is carried by both the reference and the
  generated code: keypoints are `(k + anchor)·stride` for yolo11 and `(2k + anchor − 0.5)·stride` for yolo26,
  and boxes xywh vs xyxy.

## Coverage (2026-09-30)

All 70 cases PASS on Spike (yolo26 max|diff| ≤ 1.7e-5, yolo11 ≤ 1.2e-4; the DFL distances of yolo11 are
larger in absolute terms, so its tolerance 1e-4 + 1e-2·max|ref| is looser); argmax matches where checked.
No case needed `HWMLIRFLAGS`. Spike cycles (`rdcycle` around `net`) and weight blob size per scale:

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

The weights total 5.9 GB; the largest case (yolo26x_p6) runs in a few minutes on Spike. Per-case
result lines are in `.logs/run_full.txt` (`../tools/run_full.sh yolo` regenerates it) and each
`<case>/README.md` (`../tools/gen_case_readme.py yolo`).
