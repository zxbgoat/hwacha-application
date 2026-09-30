#!/usr/bin/env python3
"""Export an Ultralytics YOLO11 / YOLO26 model (github.com/ultralytics/yolo11, .../yolo26, built from its
   yaml with random weights) to linalg-on-tensors MLIR via torch-mlir, with a PyTorch reference.
   usage: export_yolo.py <case> <out.mlir> <out_check.bin> [HW]
   <case> = yolo26<scale>[_<task>] (tasks: seg/sem/depth/cls/pose/obb/p2/p6) or
            yolo11<scale>[_<task>] (tasks: seg/cls/pose/obb; YOLO11 has no sem/depth/p2/p6 and is not
            end2end); scale in n/s/m/l/x, no task = detection; HW = input size (default 64)."""
import os, sys, re, struct, numpy as np
os.environ.setdefault('YOLO_OFFLINE', '1'); os.environ.setdefault('YOLO_VERBOSE', 'false')
import torch
from torch_mlir import fx
import inspect, torch_mlir.extras.fx_importer as _fxi
import ultralytics.nn.tasks as T
from ultralytics.nn.modules import head as H
# torch-mlir turns every constant tensor into a literal via np.array(tensor.tolist()): one Python float
# object per element, ~30x the tensor's size (see ../torchvision/export_tv.py). Convert through numpy.
# The importer hands the array's raw buffer to DenseResourceElementsAttr, so a non-contiguous tensor (Detect
# caches its anchors as `make_anchors(...).transpose(0, 1)`) must be made contiguous first, or the constant
# lands transposed in memory (the boxes came out with every anchor's x / y coordinates permuted).
def _tensor_to_numpy(tensor, npy_dtype):
    try:
        with torch.utils._mode_utils.no_dispatch():
            a = tensor.detach().cpu().numpy()
            if a.ndim and not a.flags.c_contiguous: a = np.ascontiguousarray(a)   # 0-d stays 0-d (splat path)
            return np.asarray(a).astype(npy_dtype, copy=False)
    except Exception:
        return np.array(tensor.tolist()).astype(npy_dtype)
_src = inspect.getsource(_fxi._make_vtensor_literal_op)
assert "np.array(tensor.tolist()).astype(npy_dtype)" in _src
_fxi._tensor_to_numpy = _tensor_to_numpy
exec(_src.replace("np.array(tensor.tolist()).astype(npy_dtype)", "_tensor_to_numpy(tensor, npy_dtype)"), _fxi.__dict__)

name, out, check = sys.argv[1:4]
HW = int(sys.argv[4]) if len(sys.argv) > 4 else 64
# YOLO11's yamls live in cfg/models/11 with the same task suffixes minus sem/depth/p2/p6, and the package
# ships the v11 head classes (Segment / Pose / OBB, i.e. the non-26 variants).
TASKS = {'yolo26': 'seg|sem|depth|cls|pose|obb|p2|p6', 'yolo11': 'seg|cls|pose|obb'}
mt = re.fullmatch(r'(yolo11|yolo26)([nsmlx])(?:_(seg|sem|depth|cls|pose|obb|p2|p6))?', name)
assert mt and (mt.group(1) != 'yolo11' or mt.group(3) in (None, *TASKS['yolo11'].split('|'))), \
    f'{name}: expected yolo11|yolo26<n|s|m|l|x>[_<task>] (yolo11: {TASKS["yolo11"]})'
family, scale, task = mt.group(1), mt.group(2), mt.group(3) or ''
yaml = f'{family}{scale}{"-" + task if task else ""}.yaml'    # cfg/models/11/ for yolo11, cfg/models/26/ for yolo26
MODEL = {'': T.DetectionModel, 'p2': T.DetectionModel, 'p6': T.DetectionModel, 'seg': T.SegmentationModel,
         'sem': T.SemanticSegmentationModel, 'depth': T.DepthModel, 'cls': T.ClassificationModel,
         'pose': T.PoseModel, 'obb': T.OBBModel}[task]
torch.manual_seed(0)
m = MODEL(yaml, verbose=False)
# Untrained BN sits at its zero-mean / unit-var fixed point, which drives the outputs to ~0; give BN random
# running stats (as in ../torchvision). The heads' final 1x1 convs are the only convs with a bias, and
# Detect.bias_init sets the class bias to log(5/nc/(640/stride)^2) ~ -11.5, i.e. sigmoid scores ~1e-5 that
# the tolerance could never see; draw every conv bias at random instead. The same weights feed the PyTorch
# reference and the Hwacha build, so the case still tests exact agreement.
for mod in m.modules():
    if isinstance(mod, torch.nn.BatchNorm2d):
        mod.running_mean.normal_(0, 0.1); mod.running_var.uniform_(0.5, 1.5)
        mod.weight.data.uniform_(0.5, 1.5); mod.bias.data.normal_(0, 0.1)
    elif isinstance(mod, torch.nn.Conv2d) and mod.bias is not None:
        mod.bias.data.normal_(0, 1.0)
m.eval()
head = m.model[-1]
cls_scores = False
if isinstance(head, H.Detect):
    # The decoded detection output is (B, 4 + nc [+ nm mask coefficients | nk keypoints | 1 angle], anchors)
    # in pixels and sigmoid class scores. YOLO26 is NMS-free and infers from the one2one branch (end2end);
    # YOLO11 has no one2one branch, so it infers from one2many. Either way Detect.postprocess (top max_det
    # anchors by score, a topk+gather) is replaced by the identity: the selection is left to the host, as in
    # ../torchvision's detection cases, and the exported network returns every anchor. export=True drops the
    # (predictions, raw dict) tuple of the eval path. Boxes: yolo26 reg_max=1 (no DFL) decodes to xyxy,
    # yolo11 reg_max=16 runs the DFL softmax over the 16 bins and decodes to xywh.
    if head.end2end: head.end2end = True      # yolo26: keep the one2one inference head (no-op otherwise)
    head.export = True
    H.Detect.postprocess = lambda self, p: p
elif isinstance(head, H.Classify):
    cls_scores = True          # eval returns (softmax, logits): keep the logits, the host checks their argmax

class Wrap(torch.nn.Module):
    """One input tensor in, one flat float tensor out (every output flattened and concatenated, in order):
       Segment26 returns (predictions, prototype masks), Classify (softmax, logits)."""
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x):
        y = s.m(x)
        if cls_scores: return y[1]
        ys = list(y) if isinstance(y, (tuple, list)) else [y]
        return torch.cat([t.flatten(1) for t in ys], 1)
w = Wrap(m)
x = torch.randn(1, 3, HW, HW)
with torch.no_grad(): ref = w(x); raw = m(x)
shapes = [tuple(t.shape) for t in (raw if isinstance(raw, (tuple, list)) else [raw])]
mod = fx.export_and_import(w, x, output_type='linalg-on-tensors', func_name='net')
with open(out, 'wb') as f:
    mod.operation.print(file=f, binary=True)
with open(check, 'wb') as f:
    f.write(struct.pack('i', 1 if cls_scores else 0))        # 1: class scores, the host also compares the argmax
    f.write(struct.pack('i', x.numel())); f.write(x.numpy().astype(np.float32).tobytes())
    f.write(struct.pack('i', ref.numel())); f.write(np.ascontiguousarray(ref).astype(np.float32).tobytes())
print("exported", name, yaml, type(head).__name__, "input", list(x.shape), "outputs", shapes, "->", list(ref.shape),
      "params %.1fM" % (sum(p.numel() for p in m.parameters()) / 1e6))
