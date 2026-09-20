# torchvision classification models on Hwacha

Every classification model of docs.pytorch.org/vision/stable/models.html, one directory per model,
run through PyTorch -> torch-mlir (linalg on tensors) -> hwacha-mlir -> hwacha-cc and checked on Spike
against PyTorch's own forward (random weights, fixed seed). Each `<model>/` is self-contained:

| file | what |
|---|---|
| `<model>_tv.s` | hwacha-cc assembly of the network (`net`) |
| `<model>_tv_weights.bin` + `.bin.S` | the parameters, stripped to a blob and pulled in with `.incbin` |
| `<model>_tv_check.bin` | the input and PyTorch's output, embedded by the host with `.incbin` |
| `tv_main.c` | generic host: runs `net`, compares max\|diff\| against a tolerance and the argmax |
| `hwlib.s` | conv / pool / norm library kernels every model links |
| `HWMLIRFLAGS` | present only when the case needed `--collapse-all` (hwacha-cc ran out of registers otherwise) |

The weight blobs (`*_weights.bin`) are not in git (21 GB); after a fresh clone run `make gen-<model>` (or
`make gen-all`) to recreate them -- the export uses a fixed seed, so the regenerated assembly and
weights are byte-identical to the committed ones.

`make` builds all, `make run` runs all on Spike, `make <model>` / `make <model>.spike` for one.
`MEM=<MB>` (default 2048) sets the simulator memory; the host's arena is everything above the loaded
image, so a model that prints `arena overflow` just needs a larger `MEM`.

`make gen-<model>` regenerates a case from PyTorch (`gen.sh`: export_tv.py -> mlir-opt bufferize ->
hwacha-mlir --weights-bin -> hwacha-cc, retrying with `--collapse-all` when hwacha-cc runs out of
registers); `models.txt` holds each model's input size and constructor kwargs, `make gen-all` does every
model that has no assembly yet. It needs the hwacha-cc build tree, mlir-opt and the torch-mlir venv
(`../.tmenv`, symlinked from `/tmp/tmenv`).

## Status (2026-09-20)

All 80 classification models of torchvision 0.24 build and PASS on Spike (`make run`): 21 architectures,
squeezenet1_0 .. regnet_y_128gf / vit_h_14. Largest cases: regnet_y_128gf and vit_h_14 carry 2.5 GB of
weights each and take 7-11 minutes on Spike; the whole `make run` is about 1.5 hours. The directory is
43 GB with the .riscv images, 22 GB after `make clean`.

The quantized, detection, segmentation, keypoint, video and optical-flow model families of the same
docs page are not covered: they need int8 kernels, multi-output / dynamic-shape graphs or 5-D video
inputs that the linalg-on-tensors -> hwacha-mlir route does not handle today.

## Input sizes

Inputs are 1x3x32x32 unless the trunk cannot take it: alexnet 64 (its last max-pool needs >=2x2),
inception_v3 80 (docs minimum 75), the ViTs `image_size=64` (56 for vit_h_14, patch 14), maxvit_t 64.
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
- hwacha-mlir (`mlir/hwacha-mlir.cpp`, `flattenGlobals`) flattens multi-dimensional constant globals
  to 1-D before LLVM translation: MLIR otherwise builds one `ConstantDataArray` per innermost row, and
  1x1 conv weights (innermost dim 1) turned efficientnet_v2_l's 112M floats into 12 GB of heap.
