# torchvision.transforms.v2 on Hwacha

The image transforms of docs.pytorch.org/vision/stable/transforms.html (v2 API, torchvision 0.24), one
case per transform, run through PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc and checked on Spike
against PyTorch (fixed seed). Layout, generic host, `gen.sh` and Makefile are those of `../torchintf`;
the cases live in `export_tvi.py`. Every case is a single-input module `net(x)` on a 3x16x16 float image
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
