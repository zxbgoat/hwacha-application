#!/usr/bin/env python3
"""Export an Ultralytics YOLOv3 / YOLOv5 / YOLOv8 / YOLOv10 / YOLO11 / YOLO26 model (github.com/ultralytics/yolov3,
   .../yolov5, .../yolov8, .../yolov10, .../yolo11, .../yolo26, built from its yaml with random weights) to
   linalg-on-tensors MLIR via torch-mlir, with a PyTorch reference.
   usage: export_yolo.py <case> <out.mlir> <out_check.bin> [HW]
   <case> = yolov3[_spp|_tiny] (Darknet-53 detector; v3 has no n/s/m/l/x scales, only these three yamls),
            yolov10<scale> (v10Detect, NMS-free; detection only -- the package ships six yolov10{n,s,m,l,b,x} yamls, no task variants),
            yolov12<scale> (github.com/sunsmarterjie/yolov12: A2C2f area-attention detector, built with the fork at ../.yolov12; detection only),
            yolo26<scale>[_<task>] (tasks: seg/sem/depth/cls/pose/obb/p2/p6),
            yolo11<scale>[_<task>] (tasks: seg/cls/pose/obb; YOLO11 has no sem/depth/p2/p6 and is not end2end),
            yolov8<scale>[_<task>] (tasks: seg/cls/pose/obb/p2/p6/seg_p6/pose_p6; not end2end) or
            yolov5<scale>[_<task>] (task: p6 only -- YOLOv5 ships detection only; not end2end);
            scale in n/s/m/l/x, no task = detection; HW = input size (default 64)."""
import os, sys, re, struct, numpy as np
os.environ.setdefault('YOLO_OFFLINE', '1'); os.environ.setdefault('YOLO_VERBOSE', 'false')
# YOLOv12 (github.com/sunsmarterjie/yolov12) is a fork of ultralytics 8.3.63 whose area-attention blocks differ
# from the AAttn / ABlock / A2C2f that upstream 8.4.x ships under the same names (separate qk / v convs and a
# 5x5 positional-encoding conv vs a fused qkv and a 7x7 one; mlp_ratio 1.5 vs 1.2 for the l / x scales), so the
# yolov12 cases are built with the fork itself: a checkout at ../.yolov12 (commit 2abab71; `git clone --depth 1
# https://github.com/sunsmarterjie/yolov12 ../.yolov12`) put in front of sys.path before ultralytics is imported.
if sys.argv[1:2] and sys.argv[1].startswith('yolov12'):
    _fork = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.yolov12'))
    assert os.path.isdir(os.path.join(_fork, 'ultralytics')), \
        f'yolov12 needs the fork checkout: git clone --depth 1 https://github.com/sunsmarterjie/yolov12 {_fork}'
    sys.path.insert(0, _fork)
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
# The yamls live in cfg/models/{v5,v8,11,26}: YOLO11 has the same task suffixes as YOLO26 minus sem/depth/p2/p6;
# YOLOv8 additionally has the P6 variants of seg / pose (yolov8-seg-p6.yaml, yolov8-pose-p6.yaml; case task
# seg_p6 / pose_p6); YOLOv5 ships detection only (yolov5.yaml, yolov5-p6.yaml). v5 / v8 / 11 use the package's
# non-26 head classes (Segment / Pose / OBB) and are not end2end.
TASKS = {'yolo26': 'seg|sem|depth|cls|pose|obb|p2|p6', 'yolo11': 'seg|cls|pose|obb',
         'yolov8': 'seg|cls|pose|obb|p2|p6|seg_p6|pose_p6', 'yolov5': 'p6', 'yolov3': 'spp|tiny',
         'yolov10': '', 'yolov12': ''}
# YOLOv3 has no scale letters: its three yamls are yolov3 / yolov3-spp / yolov3-tiny (case yolov3[_spp|_tiny]).
mt = re.fullmatch(r'yolov3(?:_([a-z0-9_]+))?', name)
if mt:
    family, scale, task = 'yolov3', '', (mt.group(1) or '')
    assert task in ('', 'spp', 'tiny'), f'{name}: expected yolov3[_spp|_tiny]'
else:
    # v10 also ships a 'b' (balanced) yaml; the other families only have n/s/m/l/x (their regex simply
    # never matches a b case).
    mt = re.fullmatch(r'(yolov5|yolov8|yolov10|yolov12|yolo11|yolo26)([nsmlxb])(?:_([a-z0-9_]+))?', name)
    assert mt and mt.group(3) in (None, *TASKS[mt.group(1)].split('|')), \
        f'{name}: expected yolov3[_spp|_tiny] | yolov5|yolov8|yolov10|yolov12|yolo11|yolo26<n|s|m|l|x|b>[_<task>] (' \
        + ', '.join(f'{k}: {v}' for k, v in TASKS.items()) + ')'
    family, scale, task = mt.group(1), mt.group(2), mt.group(3) or ''
# yaml name: yolov3[-spp|-tiny].yaml (no scale) vs <family in {v5,v8,v10,11,26}><scale>[-<task>].yaml
if family == 'yolov3': suffix = ('-' + task) if task else ''
else:                  suffix = scale + (('-' + task.replace('_', '-')) if task else '')
yaml = f'{family}{suffix}.yaml'
MODEL = {'': 'DetectionModel', 'p2': 'DetectionModel', 'p6': 'DetectionModel', 'spp': 'DetectionModel',
         'tiny': 'DetectionModel', 'seg': 'SegmentationModel',
         'seg_p6': 'SegmentationModel', 'sem': 'SemanticSegmentationModel', 'depth': 'DepthModel',
         'cls': 'ClassificationModel', 'pose': 'PoseModel', 'pose_p6': 'PoseModel', 'obb': 'OBBModel'}[task]
MODEL = getattr(T, MODEL, None)   # resolved lazily: the yolov12 fork (ultralytics 8.3.63) has no SemanticSegmentationModel / DepthModel
assert MODEL is not None, f"{yaml}: this ultralytics build has no model class for task {task!r}"
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
    # in pixels and sigmoid class scores. YOLO26 and YOLOv10 are NMS-free and infer from the one2one branch
    # (v10's v10Detect is a Detect subclass whose one2one branch exists, so its end2end property is true
    # without being set); YOLOv3 / YOLOv5 / YOLOv8 / YOLO11 have no one2one branch and infer from one2many.
    # Either way Detect.postprocess (top max_det anchors by score, a topk+gather) is replaced by the
    # identity: the selection is left to the host, as in ../torchvision's detection cases, and the exported
    # network returns every anchor. export=True drops the (predictions, raw dict) tuple of the eval path.
    # Boxes: yolo26 reg_max=1 (no DFL) decodes to xyxy, v3 / v5 / v8 / v10 / v11 reg_max=16 run the DFL
    # softmax over the 16 bins and decode to xywh.
    if head.end2end: head.end2end = True      # yolo26 / v10: keep the one2one inference head (no-op otherwise)
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
