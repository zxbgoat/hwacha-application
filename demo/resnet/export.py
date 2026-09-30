#!/usr/bin/env python3
"""Export the pretrained torchvision ResNet-50 (ImageNet-1K V2 weights) to linalg-on-tensors MLIR via
   torch-mlir, together with one preprocessed image and PyTorch's logits for it.
   usage: export.py <image> <out.mlir> <out_check.bin> <out_labels.h>
   The Hwacha host (main.c) embeds the image tensor and the reference, runs `net` and prints the top-5
   ImageNet classes; the class names go into out_labels.h."""
import sys, struct, numpy as np, torch, torchvision.models as M
from torchvision.io import decode_image
from torch_mlir import fx
import inspect, torch_mlir.extras.fx_importer as _fxi
# same importer patch as ../../torchvision/export_tv.py: constants through numpy instead of tensor.tolist()
# (and made contiguous, see ../../yolo/export_yolo.py)
def _tensor_to_numpy(tensor, npy_dtype):
    try:
        with torch.utils._mode_utils.no_dispatch():
            a = tensor.detach().cpu().numpy()
            if a.ndim and not a.flags.c_contiguous: a = np.ascontiguousarray(a)
            return np.asarray(a).astype(npy_dtype, copy=False)
    except Exception:
        return np.array(tensor.tolist()).astype(npy_dtype)
_src = inspect.getsource(_fxi._make_vtensor_literal_op)
assert "np.array(tensor.tolist()).astype(npy_dtype)" in _src
_fxi._tensor_to_numpy = _tensor_to_numpy
exec(_src.replace("np.array(tensor.tolist()).astype(npy_dtype)", "_tensor_to_numpy(tensor, npy_dtype)"), _fxi.__dict__)

image, out, check, labels = sys.argv[1:5]
weights = M.ResNet50_Weights.IMAGENET1K_V2                 # resnet50-11ad3fa6.pth, acc@1 80.9 %
model = M.resnet50(weights=weights).eval()
# the weights' own preprocessing: resize 232 (bilinear) -> center crop 224 -> [0,1] -> normalize(mean, std)
x = weights.transforms()(decode_image(image)).unsqueeze(0)  # 1x3x224x224
with torch.no_grad(): logits = model(x)
probs = logits.softmax(1)[0]; top = probs.topk(5)
cats = weights.meta['categories']
print("PyTorch top-5 for", image)
for v, i in zip(top.values, top.indices): print(f"  {v:6.3f}  {cats[i]}")

mod = fx.export_and_import(model, x, output_type='linalg-on-tensors', func_name='net')
with open(out, 'wb') as f: mod.operation.print(file=f, binary=True)
with open(check, 'wb') as f:                                 # [n_in][input floats][n_out][reference logits]
    f.write(struct.pack('i', x.numel())); f.write(x.numpy().astype(np.float32).tobytes())
    f.write(struct.pack('i', logits.numel())); f.write(logits.numpy().astype(np.float32).tobytes())
with open(labels, 'w') as f:
    f.write('// ImageNet-1K class names (torchvision ResNet50_Weights.IMAGENET1K_V2 meta["categories"])\n')
    f.write('static const char *const labels[1000] = {\n')
    for c in cats: f.write('  "%s",\n' % c.replace('\\', '\\\\').replace('"', '\\"'))
    f.write('};\n')
print("exported resnet50 input", list(x.shape), "output", list(logits.shape))
