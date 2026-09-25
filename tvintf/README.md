# torchvision.transforms.v2, torchvision.ops and torchvision.utils on Hwacha

The image transforms of docs.pytorch.org/vision/stable/transforms.html (v2 API, torchvision 0.24), one
case per transform, run through PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc and checked on Spike
against PyTorch (fixed seed). Layout, generic host, `gen.sh` and Makefile are those of `../torchintf`;
the transform cases live in `export_tvi.py`, the operator cases in `intf_ops.py`, the utility cases in `intf_utils.py`. Every case is a single-input module `net(x)` on a 3x16x16 float image
in [0, 1] (a 3x24x24 one, a pair or an 8-frame clip where the transform wants it) returning one float
tensor.

The `Random*` classes draw their parameters and their apply-or-not coin inside `forward`, which is
data dependent for torch.export even with p=1 and a degenerate range, so every random transform is
pinned to the functional op the class applies with one fixed draw (`RandomRotation(30)` ->
`F.rotate(x, 30)`, `RandomCrop(8)` -> `F.crop` at a fixed offset, `ColorJitter` -> the four
adjustments in the class' order with fixed factors ...); the per-case READMEs give the draw.

## Cases: 42 transforms (39 classes), all PASS

| group | case | class | Hwacha |
|---|---|---|---|
| geometry | resize | `Resize` | PASS, max\|diff\| 0, 1,657 周期 |
| geometry | resize_bilinear | `Resize` | PASS, max\|diff\| 0, 2,705 周期 |
| geometry | center_crop | `CenterCrop` | PASS, max\|diff\| 0, 94 周期 |
| geometry | random_crop | `RandomCrop` | PASS, max\|diff\| 0, 94 周期 |
| geometry | random_resized_crop | `RandomResizedCrop` | PASS, max\|diff\| 0, 2,100 周期 |
| geometry | random_horizontal_flip | `RandomHorizontalFlip` | PASS, max\|diff\| 0, 146 周期 |
| geometry | random_vertical_flip | `RandomVerticalFlip` | PASS, max\|diff\| 0, 171 周期 |
| geometry | pad | `Pad` | PASS, max\|diff\| 0, 191 周期 |
| geometry | pad_reflect | `Pad` | PASS, max\|diff\| 0, 812 周期 |
| geometry | random_zoom_out | `RandomZoomOut` | PASS, max\|diff\| 0, 211 周期 |
| geometry | random_rotation | `RandomRotation` | PASS, max\|diff\| 0, 5,584 周期 |
| geometry | random_rotation_nearest | `RandomRotation` | PASS, max\|diff\| 0, 2,951 周期 |
| geometry | random_affine | `RandomAffine` | PASS, max\|diff\| 0, 5,584 周期 |
| geometry | random_perspective | `RandomPerspective` | PASS, max\|diff\| 0, 5,845 周期 |
| geometry | elastic_transform | `ElasticTransform` | PASS, max\|diff\| 0, 5,281 周期 |
| geometry | five_crop | `FiveCrop` | PASS, max\|diff\| 0, 347 周期 |
| geometry | ten_crop | `TenCrop` | PASS, max\|diff\| 0, 782 周期 |
| geometry | random_iou_crop | `RandomIoUCrop` | PASS, max\|diff\| 0, 110 周期 |
| geometry | random_resize | `RandomResize` | PASS, max\|diff\| 0, 1,820 周期 |
| geometry | random_shortest_size | `RandomShortestSize` | PASS, max\|diff\| 0, 1,811 周期 |
| geometry | scale_jitter | `ScaleJitter` | PASS, max\|diff\| 0, 2,365 周期 |
| color | color_jitter | `ColorJitter` | PASS, max\|diff\| 0, 7,029 周期 |
| color | grayscale | `Grayscale` | PASS, max\|diff\| 0, 173 周期 |
| color | grayscale_3ch | `Grayscale` | PASS, max\|diff\| 0, 260 周期 |
| color | random_grayscale | `RandomGrayscale` | PASS, max\|diff\| 0, 260 周期 |
| color | rgb | `RGB` | PASS, max\|diff\| 0, 101 周期 |
| color | random_channel_permutation | `RandomChannelPermutation` | PASS, max\|diff\| 0, 249 周期 |
| color | random_photometric_distort | `RandomPhotometricDistort` | PASS, max\|diff\| 0, 7,029 周期 |
| color | random_adjust_sharpness | `RandomAdjustSharpness` | PASS, max\|diff\| 0, 758 周期 |
| color | random_autocontrast | `RandomAutocontrast` | PASS, max\|diff\| 0, 1,511 周期 |
| color | random_equalize | `RandomEqualize` | PASS, max\|diff\| 0, 637,725 周期 |
| color | random_invert | `RandomInvert` | PASS, max\|diff\| 0, 77 周期 |
| color | random_posterize | `RandomPosterize` | PASS, max\|diff\| 0, 262 周期 |
| color | random_solarize | `RandomSolarize` | PASS, max\|diff\| 0, 196 周期 |
| color | gaussian_blur | `GaussianBlur` | PASS, max\|diff\| 0, 2,182 周期 |
| color | gaussian_noise | `GaussianNoise` | PASS, max\|diff\| 0, 127 周期 |
| color | random_erasing | `RandomErasing` | PASS, max\|diff\| 0, 132 周期 |
| color | normalize | `Normalize` | PASS, max\|diff\| 0, 182 周期 |
| color | linear_transformation | `LinearTransformation` | PASS, max\|diff\| 0, 12,514 周期 |
| batch / video | cutmix | `CutMix` | PASS, max\|diff\| 0, 818 周期 |
| batch / video | mixup | `MixUp` | PASS, max\|diff\| 0, 749 周期 |
| batch / video | uniform_temporal_subsample | `UniformTemporalSubsample` | PASS, max\|diff\| 0, 688 周期 |

Not cases: the containers (`Compose`, `RandomApply`, `RandomChoice`, `RandomOrder`, `Identity`, `Lambda`),
the type / PIL conversions (`ToImage`, `ToPILImage`, `PILToTensor`, `ToTensor`, `ToPureTensor`, `ToDtype`,
`ConvertImageDtype`), the box / keypoint transforms (`ClampBoundingBoxes`, `ClampKeyPoints`,
`ConvertBoundingBoxFormat`, `SanitizeBoundingBoxes`, `SanitizeKeyPoints`, `SetClampingMode`), the
policy-sampling augmentations (`AutoAugment`, `RandAugment`, `TrivialAugmentWide`, `AugMix`: their
building blocks are the cases above) and `JPEG` (a codec).

**Exports that differ from the reference** (the reference is always the genuine torchvision call,
asserted equal before export): `adjust_hue` (in `color_jitter` / `random_photometric_distort`) fails
inside torch-mlir's lowering ('arith.cmpi' operands of different types), so the RGB <-> HSV
conversion is rewritten without in-place ops, aminmax and gather (one-hot selects); `autocontrast`
exports with tm_tensor.scan / scatter (index assignment) and is written as (x - min) / (max - min);
`equalize`'s scatter_add histogram crashes the export and is rewritten as torchvision's algorithm with
one-hot counting, a triangular-matmul cumulative histogram, the step rule for the LUT and a one-hot
lookup; `RandomPerspective`'s eight coefficients (aten.linalg_lstsq has no lowering) are solved on
the host and passed as `coefficients`. hwacha-cc gained a fix from this batch: `uitofp` of a sub-word
integer (posterize's uint8 path: fptosi to i8, and, uitofp) zero-extends its sign-extended source.

## torchvision.ops: 41 cases, all PASS

The operators of docs.pytorch.org/vision/stable/ops.html: the box utilities and losses on 8 boxes, the
RoI operators on a 1x4x16x16 (or 1x16x16x16 position-sensitive) map with 3 constant RoIs and a 2x2
output, deformable convolution on the same map, and the layers in eval mode.

| group | case | ops entry | Hwacha |
|---|---|---|---|
| boxes | box_area | `box_area` | PASS, max\|diff\| 0, 201 周期 |
| boxes | box_convert | `box_convert` | PASS, max\|diff\| 0, 578 周期 |
| boxes | box_iou | `box_iou` | PASS, max\|diff\| 0, 832 周期 |
| boxes | generalized_box_iou | `generalized_box_iou` | PASS, max\|diff\| 0, 1,248 周期 |
| boxes | distance_box_iou | `distance_box_iou` | PASS, max\|diff\| 0, 2,006 周期 |
| boxes | complete_box_iou | `complete_box_iou` | PASS, max\|diff\| 0, 2,928 周期 |
| boxes | clip_boxes_to_image | `clip_boxes_to_image` | PASS, max\|diff\| 0, 287 周期 |
| boxes | remove_small_boxes | `remove_small_boxes` | PASS, max\|diff\| 0, 353 周期 |
| boxes | masks_to_boxes | `masks_to_boxes` | PASS, max\|diff\| 0, 2,889 周期 |
| boxes | nms | `nms` | PASS, max\|diff\| 0, 2,310 周期 |
| boxes | batched_nms | `batched_nms` | PASS, max\|diff\| 0, 2,376 周期 |
| losses | sigmoid_focal_loss | `sigmoid_focal_loss` | PASS, max\|diff\| 0, 1,396 周期 |
| losses | generalized_box_iou_loss | `generalized_box_iou_loss` | PASS, max\|diff\| 0, 1,862 周期 |
| losses | distance_box_iou_loss | `distance_box_iou_loss` | PASS, max\|diff\| 0, 2,492 周期 |
| losses | complete_box_iou_loss | `complete_box_iou_loss` | PASS, max\|diff\| 0, 3,163 周期 |
| RoI operators | roi_align | `roi_align` | PASS, max\|diff\| 0, 19,626 周期 |
| RoI operators | roi_align_aligned | `roi_align` | PASS, max\|diff\| 0, 19,543 周期 |
| RoI operators | roi_align_module | `RoIAlign` | PASS, max\|diff\| 0, 19,626 周期 |
| RoI operators | multi_scale_roi_align | `MultiScaleRoIAlign` | PASS, max\|diff\| 0, 19,627 周期 |
| RoI operators | roi_pool | `roi_pool` | PASS, max\|diff\| 0, 3,216 周期 |
| RoI operators | roi_pool_module | `RoIPool` | PASS, max\|diff\| 0, 3,216 周期 |
| RoI operators | ps_roi_align | `ps_roi_align` | PASS, max\|diff\| 0, 39,555 周期 |
| RoI operators | ps_roi_align_module | `PSRoIAlign` | PASS, max\|diff\| 0, 39,555 周期 |
| RoI operators | ps_roi_pool | `ps_roi_pool` | PASS, max\|diff\| 0, 5,458 周期 |
| RoI operators | ps_roi_pool_module | `PSRoIPool` | PASS, max\|diff\| 0, 5,458 周期 |
| deformable convolution | deform_conv2d | `deform_conv2d` | PASS, max\|diff\| 0, 14,094 周期 |
| deformable convolution | deform_conv2d_mask | `deform_conv2d` | PASS, max\|diff\| 0, 14,627 周期 |
| deformable convolution | deform_conv2d_module | `DeformConv2d` | PASS, max\|diff\| 0, 14,094 周期 |
| layers | conv2d_norm_activation | `Conv2dNormActivation` | PASS, max\|diff\| 0, 1,436 周期 |
| layers | conv3d_norm_activation | `Conv3dNormActivation` | PASS, max\|diff\| 0, 9,792 周期 |
| layers | frozen_batch_norm2d | `FrozenBatchNorm2d` | PASS, max\|diff\| 0, 438 周期 |
| layers | mlp | `MLP` | PASS, max\|diff\| 0, 678 周期 |
| layers | permute | `Permute` | PASS, max\|diff\| 0, 143 周期 |
| layers | squeeze_excitation | `SqueezeExcitation` | PASS, max\|diff\| 0, 2,711 周期 |
| layers | feature_pyramid_network | `FeaturePyramidNetwork` | PASS, max\|diff\| 0, 2,993 周期 |
| layers | drop_block2d | `DropBlock2d` | PASS, max\|diff\| 0, 73 周期 |
| layers | drop_block3d | `DropBlock3d` | PASS, max\|diff\| 0, 75 周期 |
| layers | stochastic_depth | `StochasticDepth` | PASS, max\|diff\| 0, 73 周期 |
| layers | drop_block2d_fn | `drop_block2d` | PASS, max\|diff\| 0, 73 周期 |
| layers | drop_block3d_fn | `drop_block3d` | PASS, max\|diff\| 0, 75 周期 |
| layers | stochastic_depth_fn | `stochastic_depth` | PASS, max\|diff\| 0, 73 周期 |

**Exports that differ from the reference** (asserted equal before export): the C++ operators
(`nms`, `roi_align`, `roi_pool`, `ps_roi_align`, `ps_roi_pool`, `deform_conv2d`) have no torch-mlir
lowering, so the exported graphs are fixed-size tensor compositions: NMS ranks the boxes by score with
a permutation matrix and unrolls the greedy suppression over the 8 boxes (its result is the keep mask in
input order, which is what the kept-index list carries); `remove_small_boxes` returns the keep mask and
`masks_to_boxes` takes min / max of the masked coordinate grids (both return index-like, data-dependent
sizes otherwise); `roi_align` and `MultiScaleRoIAlign` follow the CUDA kernel's sampling (sr x sr
bilinear samples per bin, border clamping) with one-hot gathers, torchvision's own pure-tensor
`_roi_align` failing to lower; `roi_pool` takes the max over constant per-bin pixel masks (the RoIs are
constants); `ps_roi_align` is the position-sensitive variant of the sampling; `ps_roi_pool` and
`deform_conv2d` reuse `../deformable/dcn_ops.py`; the three IoU losses are written with `torch.where`
instead of torchvision's masked assignment (a lowering failure). hwacha-cc gained an `atanf` expansion
(complete IoU) from this batch.

## torchvision.utils: 7 cases, all PASS

The utilities of docs.pytorch.org/vision/stable/utils.html: `make_grid` (plain and normalized),
`save_image` (the quantised pixels that go into the PNG, the reference reading the written file back)
and the visualisation functions on a 16x16 uint8 image.

| case | utils entry | Hwacha |
|---|---|---|
| make_grid | `make_grid` | PASS, max\|diff\| 0, 509 周期 |
| make_grid_normalize | `make_grid` | PASS, max\|diff\| 3e-06, 1,931 周期 |
| save_image | `save_image` | PASS, max\|diff\| 0, 834 周期 |
| draw_bounding_boxes | `draw_bounding_boxes` | PASS, max\|diff\| 0, 3,420 周期 |
| draw_segmentation_masks | `draw_segmentation_masks` | PASS, max\|diff\| 0, 1,120 周期 |
| draw_keypoints | `draw_keypoints` | PASS, max\|diff\| 0, 1,804 周期 |
| flow_to_image | `flow_to_image` | PASS, max\|diff\| 0, 5,192 周期 |

**Exports that differ from the reference** (asserted equal before export): the draw_* functions
rasterise with PIL (or assign through boolean indices), so the exported graphs re-implement the
rasterisation on coordinate grids: rectangle outlines of a given width, filled circles (PIL's ellipse
of the (2r + 1)-pixel bbox, matched pixel for pixel at the case's radius 2), mask overlays with the
overlap rule and the alpha blend; `flow_to_image` normalises by the maximum norm, gets its angle from
atan with quadrant fix-ups (no atan2 lowering) and looks the colour wheel up by one-hot;
`make_grid(normalize=True)` clamps with the tensor's own min / max as python numbers (data
dependent), rewritten as the same per-image normalisation in tensor ops.
