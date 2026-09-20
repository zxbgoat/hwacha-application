# torch.nn.functional on Hwacha

Every function of docs.pytorch.org/docs/2.14/nn.functional.html as its own case, one directory per
function, run through PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc and checked on Spike against
PyTorch (fixed seed). Each `<fn>/` holds `<fn>.s` (the assembly of `net`), `mod_main.c` (the generic
host: `net(float* x)` vs the reference with a relative tolerance), `<fn>_check.bin` (input + PyTorch
output, embedded with `.incbin`), `HWMLIRFLAGS` (the hwacha-mlir mapping the case needed) and, for the
six conv / pool cases that call the library kernels, `hwlib.s`.

`make` builds all (`<fn>/<fn>.riscv`), `make run` runs all on Spike, `make <fn>` / `make <fn>.spike`
for one, `make gen-<fn>` regenerates a case from PyTorch (`export_function.py`, needs the hwacha-cc
build tree, mlir-opt and the torch-mlir venv `../.tmenv`), `make gen-all` every missing one.

Every case is a single-input module `net(x)`: weights, second operands and loss targets are constant
buffers, in-place variants (`relu_` ...) run on a copy, dropout runs in eval mode, losses return their
mean as a 1-element tensor, integer-index functions (`embedding`, `one_hot`) take the float input cast
to long. Inputs are small (a 1x16 row for the activations, 1x4x8 / 1x4x8x8 / 1x4x4x4x4 feature maps).

## Coverage: 111 of 118 functions, all PASS

| section | cases |
|---|---|
| convolution | conv1d/2d/3d, conv_transpose1d/2d/3d, unfold, fold |
| pooling | avg/max/lp/adaptive_avg/adaptive_max pool 1d/2d/3d, max_unpool1d/2d/3d |
| attention | scaled_dot_product_attention |
| activations | all 33 of the section incl. the in-place variants, gumbel_softmax, prelu, rrelu (eval) |
| normalization | batch_norm, group_norm, instance_norm, layer_norm, local_response_norm, rms_norm, normalize |
| linear / dropout | linear, bilinear; dropout, alpha_dropout, feature_alpha_dropout, dropout1d/2d/3d |
| sparse / distance | embedding, embedding_bag, one_hot; pairwise_distance, cosine_similarity, pdist |
| losses | 20 of 22 (all but ctc_loss and linear_cross_entropy) |
| vision | pixel_shuffle, pixel_unshuffle, pad, interpolate, upsample, upsample_nearest, upsample_bilinear, grid_sample, affine_grid |

Six functions torch-mlir / torch.export cannot take directly are exported as an equivalent op
composition; the reference in their `check.bin` still comes from the genuine `F.<fn>` call
(`export_function.py` asserts both agree before exporting): `fold` (non-overlapping: reshape+permute;
the op lowers to `tm_tensor.scatter`), `max_unpool1d/2d/3d` (x kept where it equals its window max;
the unpool is a scatter), `embedding_bag` (mean of the embeddings; no lowering), `pdist` (upper
triangle of the pairwise norms; `aten._pdist_forward` has no lowering), `gaussian_nll_loss` (the
closed form; F's checks on `var` are data-dependent), `multilabel_margin_loss` (closed form with the
target sets as float masks -- see known issues).

Not covered:
- `ctc_loss`: `aten._ctc_loss` has a data-dependent output shape, torch.export refuses it.
- `fractional_max_pool2d/3d`: `aten.fractional_max_pool2d` is marked illegal in torch-mlir.
- `linear_cross_entropy`, `grouped_mm`, `scaled_mm`, `scaled_grouped_mm`: not in torch 2.9.1 (the
  installed version; the 2.14 docs list them), `data_parallel` is a torch.nn.parallel entry.

## Known issues

`known-issues/multilabel_margin_loss_decomposed.mlir` is torch-mlir's decomposition of
`aten.multilabel_margin_loss` (index / min-with-index / gather / masked outer-product generics). It
compiles, but the Hwacha result is wrong (5.88 vs 1.52), while each of its constructs re-expressed in
PyTorch (argmin, gather, i1 any-reduction, masked select, two-dim reduction) and a step-by-step
reconstruction of the graph all match PyTorch. Not yet bisected at the MLIR level.

hwacha-cc fix that came out of this directory: `uitofp`/`sitofp` of an i1 (a predicate register)
emitted an illegal `vfcvt` (hwacha-compiler 76ad080).
