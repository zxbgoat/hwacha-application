# torchvision models on Hwacha

The models of docs.pytorch.org/vision/stable/models.html, one directory per model, run through
PyTorch -> torch-mlir (linalg on tensors) -> hwacha-mlir -> hwacha-cc and checked on Spike against
PyTorch's own forward (random weights, fixed seed). Each `<model>/` is self-contained:

| file | what |
|---|---|
| `<model>_tv.s` | hwacha-cc assembly of the network (`net`) |
| `<model>_tv_weights.bin` + `.bin.S` | the parameters, stripped to a blob and pulled in with `.incbin` |
| `<model>_tv_check.bin` | the input and PyTorch's output, embedded by the host with `.incbin` |
| `tv_main.c` | generic host: runs `net`, compares max\|diff\| against a tolerance (and the argmax for class scores) |
| `hwlib.s` | conv / pool / norm library kernels every model links |
| `HWMLIRFLAGS` | present when the case needed `--collapse-all` / `--unroll-small=2` (hwacha-cc ran out of registers otherwise) |

The weight blobs (`*_weights.bin`) are not in git (21 GB); after a fresh clone run `make gen-<model>` (or
`make gen-all`) to recreate them -- the export uses a fixed seed, so the regenerated assembly and
weights are byte-identical to the committed ones.

`make` builds all, `make run` runs all on Spike, `make <model>` / `make <model>.spike` for one.
`MEM=<MB>` overrides the simulator memory (by default sized per case from its weights); the host's arena
is everything above the loaded image.

`make gen-<model>` regenerates a case from PyTorch (`gen.sh`: export_tv.py -> mlir-opt bufferize ->
hwacha-mlir --weights-bin -> hwacha-cc, retrying with `--collapse-all` and `--unroll-small=2` when
hwacha-cc runs out of registers); `models.txt` holds each model's input size and constructor / export
options, `make gen-all` does every model that has no assembly yet. It needs the hwacha-cc build tree,
mlir-opt and the torch-mlir venv (`../.tmenv`, symlinked from `/tmp/tmenv`).

## Coverage (2026-09-20)

| family | models | what is compared |
|---|---|---|
| classification | 80 (all of torchvision 0.24) | logits (1000 classes) + argmax |
| semantic segmentation | 6: deeplabv3 x3, fcn x2, lraspp | the `out` logits map (1x21xHxW) |
| object detection | 12: Faster/Mask/Keypoint R-CNN, FCOS, RetinaNet, SSD, SSDlite | the network part: backbone + FPN + heads over every anchor (two-stage models up to the RPN head); the post-processing (score threshold, NMS) has data-dependent shapes that torch.export cannot make static, so it is left to the host |
| video classification | 7: r3d_18, mc3_18, r2plus1d_18, s3d, swin3d t/s/b | logits (400 classes) on a 1x3xTxHxW clip |
| optical flow | 2: raft_small, raft_large | the last refined flow (1x2xHxW) after `_iters` updates on a 2x3xHxW frame pair |

Not covered: the 12 **quantized** models (torch.export does not take eager-mode quantized modules: the
packed `Conv2dPackedParamsBase` weights have no `__obj_flatten__`), and **mvit_v1_b / mvit_v2_s** (a
grouped 3-D convolution that torch-mlir marks illegal, `aten.convolution` with groups on 5-D input).

Largest cases: regnet_y_128gf and vit_h_14 carry 2.5 GB of weights each and take 7-11 minutes on
Spike; ssd300_vgg16 (300x300 input) takes 22 minutes. The directory is ~50 GB with the .riscv images.

## Input sizes

Inputs are 1x3x32x32 unless the trunk cannot take it: alexnet 64 (its last max-pool needs >=2x2),
inception_v3 80 (docs minimum 75), the ViTs `image_size=64` (56 for vit_h_14, patch 14), maxvit_t 64,
detection 64 (300 for ssd300_vgg16, whose head is sized for it), RAFT 128 (feature maps are /8 and the
correlation pyramid needs >=16), video clips T=8 (16 for s3d and swin3d) at 32x32 (64 for s3d, whose
head is replaced by a global average pool since its AvgPool3d((2,7,7)) is sized for 224x224).
maxvit_t is built directly as `MaxVit(...)` with the maxvit_t block configuration but `partition_size=2`
instead of 7, since 7 only divides the stage grids of a 224x224 input.

The host tolerates inputs up to 256x256 (`in[]` in tv_main.c).

## Export notes

- `export_tv.py` patches torch-mlir's fx importer to hand constant tensors to numpy directly instead of
  going through `tensor.tolist()` (one Python object per element: the 100M+ parameter models ran the
  machine out of memory), and streams the module to disk instead of building one giant string.
- Swin: the relative-position bias `table[index]` uses a registered index buffer that the importer
  rejects; in eval it is a constant, so it is precomputed per block and exported as a plain buffer.
- ViT: torchvision zero-initializes the classification head, which makes every logit 0; any all-zero
  Linear gets a small random init so the comparison means something.
- hwacha-cc (`src/CodeGen.cpp`): a uniform `load i1` (the ViT attention masks) goes through one
  per-kernel vs temp, since a shared load cannot target a vector register.
- Weights over 2 GB (regnet_y_128gf, vit_h_14) exceed the reach of pc-relative addressing, so
  `split_weights.py` puts the first half of the blob in `.weights_lo` before the code and the second
  half in `.weights_hi` after it (`tv.ld`); the host takes its arena start from an absolute word.
- swin3d: `torch.roll` with a zero shift on a dim (the temporal window spans the clip) lowers to a
  0-sized slice + concat that bufferizes into an out-of-bounds subview; the export drops zero-shift dims.
- hwacha-mlir: the stride-2 depthwise conv library kernel was called with pad=K-1 instead of 2, so every
  5x5/7x7 stride-2 depthwise conv read its window 2 pixels off. mobilenet_v3 / mnasnet / efficientnet
  "passed" anyway because their untrained logits are ~1e-2 and the 1e-3 absolute tolerance hid it (the
  lraspp segmentation map did not); the tolerance floor is now 1e-4.
- hwacha-cc: an indexed (gather) access whose uniform base ends up in a vector register (defined under
  divergent control flow) folds the base into the index, since vlx/vsx need a vs base (FPN upsampling).
- hwacha-mlir (`mlir/hwacha-mlir.cpp`, `flattenGlobals`) flattens multi-dimensional constant globals
  to 1-D before LLVM translation: MLIR otherwise builds one `ConstantDataArray` per innermost row, and
  1x1 conv weights (innermost dim 1) turned efficientnet_v2_l's 112M floats into 12 GB of heap.
