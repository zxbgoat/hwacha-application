# pytorchvideo model zoo on Hwacha

The networks of pytorchvideo.readthedocs.io/en/latest/model_zoo.html (pytorchvideo 0.1.5): the
Kinetics-400 table (C2D, I3D, Slow, SlowFast, CSN, R(2+1)D, X3D, MViT), the AVA detection models and the
accelerator zoo's EfficientX3d, one case per row (the Something-Something V2 and Charades rows are the
Kinetics Slow / SlowFast 8x8 architectures with a different head size and are not repeated). Every case
is the TorchHub builder's architecture (`pytorchvideo.models.hub`) at full width and depth with random
weights (fixed seed), run through PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc and checked on
Spike against PyTorch. Layout, host (`tv_main.c`, weights outside the assembly via `--weights-bin`),
`tv.ld` / `split_weights.py` and Makefile are those of `../torchvision`; the cases live in `export_tvid.py`.

Clips are 1x3xTx32x32 (T = the setting's frame length: 8 for 8x8, 4 for 4x16, 16 for 16x4 / 16x5 / 16x8,
13 for 13x6, 32 for 32x2 / 32x3) instead of 224x224; the detection cases take a 64x64 clip so the /16
map is 4x4. SlowFast takes both pathways in one tensor, the T slow frames followed by the 4T fast ones.
The heads sized for 224x224 (an `AvgPool3d((T', 7, 7))` on the 7x7 map after /32, i.e. a global average)
are replaced by `AdaptiveAvgPool3d(1)`; X3D is built through `create_x3d` with `input_crop_size=32`,
from which the builder sizes its head pool (the hub's `x3d_*` fix it at 160 / 224 / 312).

## Cases: 20, all PASS

| group | case | model zoo row | params | Hwacha |
|---|---|---|---|---|
| Kinetics-400 | c2d_r50 | C2D R50 8x8 | 24.33M | PASS, max\|diff\| 0.00013, 1,131,792,781 周期 |
| Kinetics-400 | i3d_r50 | I3D R50 8x8 | 28.04M | PASS, max\|diff\| 9.5e-05, 1,702,456,306 周期 |
| Kinetics-400 | slow_r50_4x16 | Slow R50 4x16 | 32.45M | PASS, max\|diff\| 5e-05, 1,836,314,385 周期 |
| Kinetics-400 | slow_r50 | Slow R50 8x8 | 32.45M | PASS, max\|diff\| 6.7e-05, 3,364,807,806 周期 |
| Kinetics-400 | slowfast_r50_4x16 | SlowFast R50 4x16 | 34.57M | PASS, max\|diff\| 0.00013, 2,104,805,172 周期 |
| Kinetics-400 | slowfast_r50 | SlowFast R50 8x8 | 34.57M | PASS, max\|diff\| 0.00013, 3,895,505,186 周期 |
| Kinetics-400 | slowfast_r101 | SlowFast R101 8x8 | 62.83M | PASS, max\|diff\| 0.046875, 10,216,933,574 周期 |
| Kinetics-400 | slowfast_16x8_r101_50_50 | SlowFast R101_50_50 16x8 | 53.77M | PASS, max\|diff\| 0.126953, 13,347,403,205 周期 |
| Kinetics-400 | csn_r101 | CSN R101 32x2 | 22.21M | PASS, max\|diff\| 1e-06, 775,283,768 周期 |
| Kinetics-400 | r2plus1d_r50 | R(2+1)D R50 16x4 | 28.11M | PASS, max\|diff\| 1e-06, 3,162,653,849 周期 |
| Kinetics-400 | x3d_xs | X3D XS 4x12 | 3.79M | PASS, max\|diff\| 0, 44,777,884 周期 |
| Kinetics-400 | x3d_s | X3D S 13x6 | 3.79M | PASS, max\|diff\| 0, 87,244,343 周期 |
| Kinetics-400 | x3d_m | X3D M 16x5 | 3.79M | PASS, max\|diff\| 0, 96,480,635 周期 |
| Kinetics-400 | x3d_l | X3D L 16x5 | 6.15M | PASS, max\|diff\| 0, 173,578,445 周期 |
| Kinetics-400 | mvit_base_16x4 | MViT B 16x4 | 36.61M | PASS, max\|diff\| 3e-06, 95,051,219 周期 |
| Kinetics-400 | mvit_base_32x3 | MViT B 32x3 | 36.61M | PASS, max\|diff\| 2e-06, 187,781,550 周期 |
| AVA v2.2 detection | slow_r50_detection | Slow R50 4x16 (AVA) | 31.78M | PASS, max\|diff\| 1.3e-05, 6,569,853,609 周期 |
| AVA v2.2 detection | slowfast_r50_detection | SlowFast R50 8x8 (AVA) | 33.82M | PASS, max\|diff\| 1.3e-05, 14,601,039,989 周期 |
| accelerator zoo | efficient_x3d_xs | X3D_XS (fp32) | 3.79M | PASS, max\|diff\| 0, 44,768,847 周期 |
| accelerator zoo | efficient_x3d_s | X3D_S (fp32) | 3.79M | PASS, max\|diff\| 0, 87,323,204 周期 |

The two SlowFast R101 cases' logits reach 3.5e4 / 5.8e4 in magnitude (random BatchNorm statistics through
101 layers), so their max|diff| of 0.05 / 0.13 is float32 rounding at ~1e-6 relative; every other case
agrees to 1e-4 or better.

**Exports that differ from the pytorchvideo forward** (every replacement is checked against the
genuine module on the same weights before export):

- CSN, X3D, EfficientX3d: the depthwise 3-D convolutions (`groups` = channels) are `aten.convolution`
  with groups on 5-D input, which torch-mlir marks illegal. `DepthwiseConv3d` splits the kernel along
  its temporal taps: each tap is a 2-D depthwise convolution over the B·T frames (T folded into the
  batch, the frames selected with the temporal stride on the zero-padded clip) and the taps are summed;
  equal to `nn.Conv3d` to 1e-7.
- MViT: the cls token and the (spatial + temporal) positional embedding are computed from
  `nn.Parameter`s inside the forward and concatenated, which torch-mlir's importer rejects (a Parameter
  in `aten.cat`'s list). In eval both are constants: `mvit_const_pos` precomputes the full embedding
  (cls row included) and the token as buffers.
- detection: `torchvision.ops.RoIAlign` has no lowering. With fixed boxes the RoI pooling is a constant
  gather: `RoIAlignGather` precomputes each (bin, sample) point's bilinear weights on the feature map
  following the CUDA kernel (sampling_ratio=0 -> ceil(roi/7)² samples per bin, points within one pixel
  outside the map clamped to the border) into an (R·49, H·W) matrix, so the pooling is one matmul;
  equal to `RoIAlign` to 1e-5.

Random BatchNorm running statistics (as in `../torchvision`) keep the untrained forward non-degenerate.
The largest weight blob is slowfast_r101's (241 MB); the assembly stays at 3.6-17 MB per case since the
weights are not folded into it. `make gen-<case>` regenerates a case in 1-3 minutes.
