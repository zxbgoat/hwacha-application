# hwacha-application

PyTorch models and layers (through torch-mlir -> hwacha-mlir -> hwacha-cc) and OpenCL kernels
(through clang -> hwacha-cc) compiled for the Hwacha vector accelerator, each as a self-contained case (assembly + C host + reference
data) that builds with the RISC-V GNU toolchain and runs on Spike.

| directory | what | cases |
|---|---|---|
| `torchnn/` | single `torch.nn` layers | 121 |
| `torchfunc/` | `torch.nn.functional` functions | 111 |
| `torchintf/` | top-level `torch.*` tensor functions: `torch.topk`, the 22 `torch.fft` functions, the 11 `torch.signal.windows`, the 41 `torch.linalg` functions, the 56 `torch.special` functions | 131 |
| `torchvision/` | torchvision models (classification, segmentation, detection, video, optical flow) | 109 |
| `yolo/` | Ultralytics YOLOv3 (github.com/ultralytics/yolov3; detect, spp, tiny), YOLOv5 (detect + P6), YOLOv8 (.../yolov8), YOLOv10 (github.com/THU-MIG/yolov10; the n / s / m / l / x / b scales, detection only), YOLO11 (.../yolo11), YOLOv12 (github.com/sunsmarterjie/yolov12, built with its own ultralytics fork; detection) and YOLO26 (.../yolo26): the n / s / m / l / x scales of detection, instance segmentation, classification, pose and OBB (v8, 11, 26), the P2 / P6 detection variants (v8, 26, v5-p6), seg-p6 / pose-p6 (v8) and semantic segmentation / depth (26), heads exported over every anchor | 139 |
| `demo/resnet/` | ResNet-50 image classification on a real photo with the pretrained ImageNet-1K V2 weights: top-5 classes printed on Spike, logits within 5e-6 of PyTorch | 1 |
| `ttmodule/` | the torchtune 0.6 module reference (attention, transformer layers, decoder, ViT, LoRA / DoRA, fusion, losses, kv-cache utilities) | 34 |
| `ttmodel/` | the torchtune 0.6 model reference: llama2 / code llama / llama3 / 3.1 / 3.2 / 3.3 / 3.2 vision, qwen2 / 2.5, phi3 / 4, mistral, gemma / gemma2, clip, each plain and LoRA, at tiny sizes | 41 |
| `tvintf/` | torchvision.transforms.v2 image transforms (42, random ones pinned to one draw) torchvision.ops (41: boxes, NMS, losses, RoI operators, deformable convolution, layers) torchvision.utils (7: grids, drawing, flow colouring) and torchvision.io (8: a tensor baseline JPEG codec, PNG modes) | 98 |
| `torchaudio/` | the torchaudio 2.9 model classes (Conformer, Emformer, ConvTasNet, DeepSpeech, Wav2Letter, HDemucs, wav2vec2 / WavLM / HuBERT, RNN-T, SQUIM, Tacotron2, WaveRNN) at tiny sizes | 16 |
| `tafunc/` | torchaudio.transforms (spectral, masking, waveform, multichannel beamforming, RNN-T loss) + torchaudio.functional (IIR filters, filter banks, Fréchet distance) | 43 |
| `torchvideo/` | pytorchvideo model zoo: Kinetics-400 (C2D, I3D, Slow, SlowFast R50/R101, CSN, R(2+1)D, X3D XS-L, MViT-B), AVA detection (RoIAlign as a constant gather), EfficientX3d; depthwise Conv3d as 2-D taps | 20 |
| `torchgeometric/` | torch_geometric.nn (pytorch-geometric.readthedocs.io modules/nn.html): 55 convolutional layers, 26 aggregation operators, attention, normalization, pooling / unpooling, 44 models, KGE models, encodings, functional, dense layers, on one 8-node graph; static self-loop / scatter / int(max) / to_dense_batch / knn-radius replacements for torch.export | 182 |
| `torchoptim/` | torch.optim (docs 2.14 optim.html): the 16 algorithms and their variants (functional single-tensor updates unrolled on a quadratic), 15 LR schedulers, AveragedModel SWA / EMA and SWALR; the 15 torch.nn.init initializers (seed-0 draws pinned, orthogonal_ by Gram-Schmidt) | 57 |
| `torchopera/` | the operator sections of torch.html (docs 2.14): Tensors, creation, indexing / slicing / joining, random sampling, pointwise, reduction, comparison, spectral, other, BLAS / LAPACK, foreach — one case per entry incl. the in-place variants | 579 |
| `deformable/` | Deformable ConvNets (deform conv / PS-RoI ops, DeepLab, R-FCN, Faster R-CNN, FPN, each plain and deformable) | 20 |
| `rodinia/` | Rodinia 3.1 OpenCL benchmarks compiled directly by hwacha-cc, with bare-metal hosts (16 pass, 3 documented compiler limits, 2 without assembly) | 21 |

Each directory has its own Makefile and README (`make run` builds and runs everything on Spike).
Toolchain paths default to `/home/tesla/hwacha-compiler` (`ROOT=` overrides).

Not tracked: `.riscv` images and the torchvision / yolo weight blobs (`*_weights.bin`, 21 GB + 16 GB), which
`make gen-<model>` regenerates deterministically.
