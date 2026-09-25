# hwacha-application

PyTorch models and layers (through torch-mlir -> hwacha-mlir -> hwacha-cc) and OpenCL kernels
(through clang -> hwacha-cc) compiled for the Hwacha vector accelerator, each as a self-contained case (assembly + C host + reference
data) that builds with the RISC-V GNU toolchain and runs on Spike.

| directory | what | cases |
|---|---|---|
| `torchnn/` | single `torch.nn` layers | 121 |
| `torchfunc/` | `torch.nn.functional` functions | 111 |
| `torchintf/` | top-level `torch.*` tensor functions: `torch.topk`, the 22 `torch.fft` functions, the 11 `torch.signal.windows`, the 41 `torch.linalg` functions, the 56 `torch.special` functions | 131 |
| `torchvision/` | torchvision models (classification, segmentation, detection, video, optical flow) | 107 |
| `ttmodule/` | the torchtune 0.6 module reference (attention, transformer layers, decoder, ViT, LoRA / DoRA, fusion, losses, kv-cache utilities) | 34 |
| `ttmodel/` | the torchtune 0.6 model reference: llama2 / code llama / llama3 / 3.1 / 3.2 / 3.3 / 3.2 vision, qwen2 / 2.5, phi3 / 4, mistral, gemma / gemma2, clip, each plain and LoRA, at tiny sizes | 41 |
| `tvintf/` | torchvision.transforms.v2 image transforms (42, random ones pinned to one draw) torchvision.ops (41: boxes, NMS, losses, RoI operators, deformable convolution, layers) torchvision.utils (7: grids, drawing, flow colouring) and torchvision.io (8: a tensor baseline JPEG codec, PNG modes) | 98 |
| `deformable/` | Deformable ConvNets (deform conv / PS-RoI ops, DeepLab, R-FCN, Faster R-CNN, FPN, each plain and deformable) | 20 |
| `rodinia/` | Rodinia 3.1 OpenCL benchmarks compiled directly by hwacha-cc, with bare-metal hosts (16 pass, 3 documented compiler limits, 2 without assembly) | 21 |

Each directory has its own Makefile and README (`make run` builds and runs everything on Spike).
Toolchain paths default to `/home/tesla/hwacha-compiler` (`ROOT=` overrides).

Not tracked: `.riscv` images and the torchvision weight blobs (`*_weights.bin`, 21 GB), which
`make gen-<model>` regenerates deterministically.
