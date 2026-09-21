# Deformable ConvNets on Hwacha

The models of github.com/msracver/Deformable-ConvNets (Dai et al., ICCV 2017; MXNet, ResNet-101
backbone) rebuilt in PyTorch and run through torch-mlir -> hwacha-mlir -> hwacha-cc, one directory per
model, each checked on Spike against PyTorch's own forward (random weights, fixed seed, 64x64 inputs).
The directory layout, host (`tv_main.c`), linker script and Makefile are those of `../torch-vision`.

## The operators (`dcn_ops.py`)

torch-mlir cannot lower `torchvision.ops.deform_conv2d` / `ps_roi_pool` (custom ops), so both are
written with plain tensor ops and checked against torchvision to 1e-7:

- `deform_conv2d_ref`: the sampling grid is built from the offsets (and, for DCN v2, the modulation
  mask), each tap is a bilinear gather with zero padding, then an im2col matmul with the weights.
- `ps_roi_pool_ref`: position-sensitive RoI pooling with torchvision's exact bin rule
  (floor / ceil of `y1 + i * (y2-y1)/k`), evaluated on a fixed candidate window per bin and masked, so
  the shapes stay static; `torch.round` and `ceil` are expressed with `floor` (hwacha-cc lowers floorf
  only), and the channel-group selection is two `diagonal`s (an einsum with identity matrices
  lowered to a batch_matmul that hwacha-mlir unrolled into 2M instructions).
- `deform_ps_roi_pool_ref`: deformable PS-RoI pooling (paper sec. 2.3): each bin shifted by
  gamma * roi size * a predicted (dy, dx), sampled bilinearly on a 2x2 grid per bin.

## Cases (`export_dcn.py`)

| case | model | compared output |
|---|---|---|
| deform_conv, deform_conv_v2 | one 3x3 deformable conv (offsets from a 3x3 conv on the input; v2 adds the sigmoid mask) | the feature map |
| psroi_pool, deform_psroi | PS-RoI pooling of 3 fixed RoIs, k=7; the deformable one with per-bin offsets from a first pooling | the pooled (R, C, 7, 7) |
| deeplab, deeplab_dcn (+ `_voc`: 21 classes) | ResNet-101 conv5 dilated + 1x1 classifier + bilinear upsampling (the repo's DeepLab, Cityscapes 19 / VOC 21 classes); `_dcn`: deformable 3x3 convs in conv5 | the logits map 1xCx64x64 |
| rfcn, rfcn_dcn (+ `_voc` 21, `_coco` 81 classes) | R-FCN: conv5 dilated, 1x1 reduce, position-sensitive class / bbox maps, PS-RoI pooling of 3 fixed RoIs and voting; RPN head on conv5; `_dcn`: deformable conv5 + deformable PS-RoI pooling | RPN cls + box maps, the class votes and the bbox votes, concatenated |
| rcnn, rcnn_dcn (+ `_coco`) | Faster R-CNN (2fc): conv5 dilated, RoI pooling (7x7 average) of 3 fixed RoIs, fc 1024 x2, cls / box; `_dcn`: deformable conv5 + deformable RoI pooling (per-bin offsets from a fc) | RPN cls + box maps, cls and box scores |
| fpn, fpn_dcn | FPN on ResNet-101 (P2-P6) with the shared RPN head; `_dcn`: deformable conv5 | the RPN head outputs of all levels |

As in `../torch-vision`, the detectors' proposal selection / NMS are data-dependent and are not
exported: RoIs are fixed, and the RPN head outputs are part of what is compared. The repo's training
details (OHEM, Soft-NMS, multi-scale test) are test/training-time procedures without a place in a
single forward. Weights are random (the MXNet `.params` are not loaded): the test certifies that the
Hwacha forward reproduces PyTorch's, not the paper's accuracy.

`make` / `make run` / `make <case>` / `make <case>.spike` / `make gen-<case>` as in torch-vision.

## Results (2026-09-22)

All 20 cases PASS on Spike. BatchNorm running stats are calibrated by one train-mode forward on a
random batch (momentum 1): with 100+ random BN layers, random running stats blew the ResNet-101
outputs up to ~1e6 and the comparison degenerated into a relative-error test at float precision.

The deformable variants deviate more from PyTorch than the plain ones (max|diff| / max|ref| ~7e-4 vs
~3e-5). That is the sensitivity of a random, untrained network to rounding, not a Hwacha error:
PyTorch's own float32 forward deviates from its float64 forward by the same amount (deeplab_dcn
2.1e-2 in float32-vs-float64, 4.1e-2 Hwacha-vs-float32; deeplab 1.7e-3 vs 2.3e-3), because the
sampling offsets predicted by the random conv5 feed bilinear interpolation, and the single deformable
conv / PS-RoI operators on their own match to 4e-6 / 0.

| case | cycles | max\|diff\| | max\|ref\| |
|---|---|---|---|
| deform_conv | 1,096,749 | 4e-06 | 3.76 |
| deform_conv_v2 | 1,106,067 | 3e-06 | 2.29 |
| psroi_pool | 148,086 | 0 | 1.66 |
| deform_psroi | 299,888 | 0 | 1.66 |
| deeplab | 80,933,539 | 0.0023 | 56.5 |
| deeplab_dcn | 82,796,826 | 0.041 | 60.2 |
| deeplab_voc | 80,971,018 | 0.0033 | 69.2 |
| deeplab_dcn_voc | 82,834,356 | 0.023 | 40 |
| rfcn | 94,356,045 | 0.00091 | 19.2 |
| rfcn_dcn | 120,045,888 | 0.017 | 23.9 |
| rfcn_voc | 94,356,045 | 0.00091 | 19.2 |
| rfcn_dcn_voc | 120,045,888 | 0.017 | 23.9 |
| rcnn | 93,655,180 | 0.00079 | 29.8 |
| rcnn_dcn | 144,502,446 | 0.0069 | 16.4 |
| fpn | 82,988,111 | 0.0044 | 99.3 |
| fpn_dcn | 84,600,319 | 0.015 | 20.6 |
| rfcn_coco | 107,750,745 | 0.00094 | 31.4 |
| rfcn_dcn_coco | 133,080,849 | 0.0074 | 18.3 |
| rcnn_coco | 93,706,238 | 0.00086 | 38.6 |
| rcnn_dcn_coco | 144,502,447 | 0.013 | 13.6 |
