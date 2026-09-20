# hwacha-application

PyTorch models and layers compiled for the Hwacha vector accelerator through
torch-mlir -> hwacha-mlir -> hwacha-cc, each as a self-contained case (assembly + C host + reference
data) that builds with the RISC-V GNU toolchain and runs on Spike.

| directory | what | cases |
|---|---|---|
| `torch-module/` | single `torch.nn` layers | 121 |
| `torch-vision/` | torchvision classification models | 80 |

Each directory has its own Makefile and README (`make run` builds and runs everything on Spike).
Toolchain paths default to `/home/tesla/hwacha-compiler` (`ROOT=` overrides).

Not tracked: `.riscv` images and the torchvision weight blobs (`*_weights.bin`, 21 GB), which
`make gen-<model>` regenerates deterministically.
