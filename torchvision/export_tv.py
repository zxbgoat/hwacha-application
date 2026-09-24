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
# keys starting with "_" configure the export, the rest go to the model constructor:
#   _T: frames (video), _iters: flow refinement iterations (RAFT), _head: "raw" (detection, see below)
cfg = {k: kw.pop(k) for k in list(kw) if k.startswith('_')}
torch.manual_seed(0)

class Wrap(torch.nn.Module):
    """Make the forward take one tensor and return one float tensor (what tv_main.c compares)."""
    def __init__(s, m, f): super().__init__(); s.m = m; s.f = f
    def forward(s, x): return s.f(s.m, x)

family = next((f for f in ('segmentation', 'detection', 'video', 'optical_flow')
               if name in M.list_models(module=getattr(M, f))), 'classification')
if family == 'segmentation':
    # returns {'out': logits, 'aux': ...}: keep 'out' (aux_loss=False drops the aux head)
    m = Wrap(M.get_model(name, weights=None, weights_backbone=None, aux_loss=False), lambda m, x: m(x)['out'])
    x = torch.randn(1, 3, HW, HW)
elif family == 'video':
    if name.startswith('swin3d'):
        # torch.roll with a zero shift on some dim (the temporal window covers the whole clip, so its
        # shift is 0) lowers to a 0-sized slice + concat that the bufferizer turns into an out-of-bounds
        # subview. Drop the zero-shift dims from the roll; the result is the same.
        _roll = torch.roll
        def roll_nz(x, shifts, dims=()):
            if isinstance(shifts, int): shifts = (shifts,)
            if isinstance(dims, int): dims = (dims,)
            if not dims: return _roll(x, shifts)   # flattened roll, unchanged
            kept = [(s_, d_) for s_, d_ in zip(shifts, dims) if s_ % x.shape[d_] != 0]
            if not kept: return x
            return _roll(x, tuple(s_ for s_, _ in kept), tuple(d_ for _, d_ in kept))
        torch.roll = roll_nz
    v = M.get_model(name, weights=None, **kw)
    if name == 's3d':
        # S3D ends in AvgPool3d((2,7,7)), sized for the canonical 16x224x224 clip (7x7 after /32). Use the
        # global average instead (identical there) so the model runs on a 16x64x64 clip.
        v.avgpool = torch.nn.AdaptiveAvgPool3d(1)
    m = Wrap(v, lambda m, x: m(x))
    x = torch.randn(1, 3, int(cfg.get('_T', 8)), HW, HW)
elif family == 'optical_flow':
    # both frames stacked into one input (2,3,H,W); output = the last refined flow (1,2,H,W)
    n_it = int(cfg.get('_iters', 12))
    m = Wrap(M.get_model(name, weights=None, **kw), lambda m, x: m(x[0:1], x[1:2], num_flow_updates=n_it)[-1])
    x = torch.randn(2, 3, HW, HW)
elif family == 'detection':
    # The detectors' post-processing (score threshold, top-k, NMS) yields a data-dependent number of
    # boxes, which torch.export cannot make static. Export the network part: backbone (+FPN) and the
    # prediction heads over every anchor, i.e. what the post-processing consumes. Normalization /
    # resizing of GeneralizedRCNNTransform is skipped: the random input stands in for the normalized
    # image. Two-stage models (Faster/Mask/Keypoint R-CNN) stop at the RPN head: the RoI heads sit
    # behind proposal selection (NMS) and roi_align, neither exportable.
    det = M.get_model(name, weights=None, weights_backbone=None, **kw)
    def raw(m, x):
        feats = m.backbone(x)
        fl = list(feats.values()) if isinstance(feats, dict) else [feats]
        if hasattr(m, 'rpn'):                      # two-stage: RPN objectness + box deltas per level
            obj, deltas = m.rpn.head(fl)
            outs = [t.flatten(1) for t in obj] + [t.flatten(1) for t in deltas]
        else:                                     # one-stage: head outputs per anchor
            h = m.head(fl)
            outs = [h[k].flatten(1) for k in sorted(h)]
        return torch.cat(outs, 1)
    m = Wrap(det, raw)
    x = torch.randn(1, 3, HW, HW)
elif name == 'maxvit_t':
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
if family == 'classification': x = torch.randn(1, 3, HW, HW)
with torch.no_grad(): ref = m(x)
mod = fx.export_and_import(m, x, output_type='linalg-on-tensors', func_name='net')
with open(out, 'wb') as f:  # stream the IR to disk instead of materializing a multi-GB Python string
    mod.operation.print(file=f, binary=True)
with open(check, 'wb') as f:
    f.write(struct.pack('i', 1))
    f.write(struct.pack('i', x.numel())); f.write(x.numpy().astype(np.float32).tobytes())
    f.write(struct.pack('i', ref.numel())); f.write(np.ascontiguousarray(ref).astype(np.float32).tobytes())
print("exported", name, family, "input", list(x.shape), "output", list(ref.shape),
      "params %.1fM" % (sum(p.numel() for p in m.parameters()) / 1e6))
