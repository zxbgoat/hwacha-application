#!/usr/bin/env python3
"""Export a torchvision classification model to linalg-on-tensors MLIR via torch-mlir, with a PyTorch
   reference.  usage: export_tv.py <model_name> <out.mlir> <out_check.bin> [HW] [kwargs-json]
   HW = input size (default 32); kwargs-json = extra constructor kwargs, e.g. '{"image_size": 64}'."""
import sys, json, struct, numpy as np, torch, torchvision.models as M
from torch_mlir import fx
import inspect, torch_mlir.extras.fx_importer as _fxi
# torch-mlir turns every constant tensor into a literal via np.array(tensor.tolist()): one Python float
# object per element, ~30x the tensor's size. On the 100M+ parameter models that exhausts memory (it
# took the whole WSL VM down twice). Convert through numpy directly, falling back to the list path.
def _tensor_to_numpy(tensor, npy_dtype):
    try:
        with torch.utils._mode_utils.no_dispatch():
            return np.asarray(tensor.detach().cpu().numpy()).astype(npy_dtype, copy=False)   # asarray keeps 0-d
    except Exception:
        return np.array(tensor.tolist()).astype(npy_dtype)
_src = inspect.getsource(_fxi._make_vtensor_literal_op)
assert "np.array(tensor.tolist()).astype(npy_dtype)" in _src
_fxi._tensor_to_numpy = _tensor_to_numpy
exec(_src.replace("np.array(tensor.tolist()).astype(npy_dtype)", "_tensor_to_numpy(tensor, npy_dtype)"), _fxi.__dict__)
name, out, check = sys.argv[1:4]
HW = int(sys.argv[4]) if len(sys.argv) > 4 else 32
kw = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
torch.manual_seed(0)
if name == 'maxvit_t':
    # maxvit_t() hard-codes partition_size=7, which only divides the stage grids for 224x224 inputs.
    # Same architecture (stem 64, blocks [64,128,256,512] x [2,2,5,2], head_dim 32) with a partition that
    # fits the small input: input HW -> stem HW/2 -> stages HW/4 .. HW/32 must all be multiples of it.
    m = M.maxvit.MaxVit(stem_channels=64, block_channels=[64,128,256,512], block_layers=[2,2,5,2],
                        stochastic_depth_prob=0.2, head_dim=32, input_size=(HW, HW), **kw)
else:
    m = M.get_model(name, weights=None, **kw)
# torchvision zero-initializes the ViT classification head (heads.head), which makes every logit 0 and
# the test vacuous; give any all-zero Linear a small random init (same weights on both sides, as below).
for mod in m.modules():
    if isinstance(mod, torch.nn.Linear) and not mod.weight.detach().abs().sum():
        torch.nn.init.normal_(mod.weight, std=0.02)
        if mod.bias is not None: torch.nn.init.normal_(mod.bias, std=0.02)
# untrained BN sits at its zero-mean/unit-var fixed point, which drives the logits to ~0 and makes
# argmax meaningless; give BN random running stats so the forward is non-degenerate (same weights feed
# the PyTorch reference and the Hwacha build, so it still tests exact agreement).
for mod in m.modules():
    if isinstance(mod, torch.nn.BatchNorm2d):
        mod.running_mean.normal_(0, 0.1); mod.running_var.uniform_(0.5, 1.5)
        mod.weight.data.uniform_(0.5, 1.5); mod.bias.data.normal_(0, 0.1)
m.eval()
# Swin: the window-attention bias is table[relative_position_index] with the index a registered buffer,
# which torch-mlir's fx importer rejects (a constant tensor inside aten.index's list argument). In eval
# the bias is a constant, so precompute it per block and hand it over as a plain buffer instead.
for mod in m.modules():
    if isinstance(mod, M.swin_transformer.ShiftedWindowAttention):
        with torch.no_grad(): mod.register_buffer('rpb_const', mod.get_relative_position_bias().clone())
        mod.get_relative_position_bias = (lambda mod: (lambda: mod.rpb_const))(mod)
x = torch.randn(1, 3, HW, HW)
with torch.no_grad(): ref = m(x)
mod = fx.export_and_import(m, x, output_type='linalg-on-tensors', func_name='net')
with open(out, 'wb') as f:  # stream the IR to disk instead of materializing a multi-GB Python string
    mod.operation.print(file=f, binary=True)
with open(check, 'wb') as f:
    f.write(struct.pack('i', 1))
    f.write(struct.pack('i', x.numel())); f.write(x.numpy().astype(np.float32).tobytes())
    f.write(struct.pack('i', ref.numel())); f.write(np.ascontiguousarray(ref).astype(np.float32).tobytes())
print("exported", name, "input", list(x.shape), "output", list(ref.shape),
      "params %.1fM" % (sum(p.numel() for p in m.parameters()) / 1e6))
