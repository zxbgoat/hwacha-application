#!/usr/bin/env python3
"""Export one model of github.com/msracver/Deformable-ConvNets (MXNet, ResNet-101 backbone) rebuilt in
PyTorch, through torch-mlir to linalg-on-tensors, with a PyTorch reference.
usage: export_dcn.py <model> <out.mlir> <out_check.bin>
The deformable conv / PS-RoI pooling operators are dcn_ops.py (tensor-op versions numerically identical
to torchvision.ops.deform_conv2d / ps_roi_pool, which torch-mlir cannot lower). Inputs are small
(64x64 images) and weights random (fixed seed, BN with random running stats) - the test certifies the
Hwacha forward against PyTorch, not accuracy."""
import sys, struct, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as M
from torch_mlir import fx
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from dcn_ops import deform_conv2d_ref, ps_roi_pool_ref, deform_ps_roi_pool_ref
import export_tv_patch  # numpy constant path + streaming print (see torch-vision/export_tv.py)

class DeformConv(nn.Module):
    """3x3 deformable conv (DCN v1) as in the paper: the offsets are predicted by a 3x3 conv on the
    same input (zero-initialized there; random here) and the sampling is bilinear."""
    def __init__(s, cin, cout, stride=1, dilation=1, v2=False):
        super().__init__(); s.v2 = v2; s.stride = stride; s.dilation = dilation
        s.offset = nn.Conv2d(cin, 27 if v2 else 18, 3, stride=stride, padding=dilation, dilation=dilation)
        s.weight = nn.Parameter(torch.randn(cout, cin, 3, 3) * (2.0 / (cin * 9)) ** 0.5)
    def forward(s, x):
        o = s.offset(x)
        if s.v2: return deform_conv2d_ref(x, o[:, :18] * 0.5, s.weight, None, s.stride, s.dilation, s.dilation, mask=torch.sigmoid(o[:, 18:]))
        return deform_conv2d_ref(x, o * 0.5, s.weight, None, s.stride, s.dilation, s.dilation)

def resnet101_backbone(dcn, dilate_c5=True):
    """ResNet-101 (v1) up to conv5, as the repo uses it: conv5 dilated (stride 16 output) for R-FCN /
    Faster R-CNN / DeepLab; in the deformable variant the 3x3 convs of the conv5 stage (res5a-c) are
    deformable convs, as in the paper."""
    r = M.resnet101(weights=None, replace_stride_with_dilation=[False, False, dilate_c5])
    if dcn:
        for blk in r.layer4:
            c = blk.conv2
            blk.conv2 = DeformConv(c.in_channels, c.out_channels, c.stride[0], c.dilation[0])
    body = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4)
    return body, 2048

class DeepLab(nn.Module):
    """DeepLab (ASPP-free, as in the repo's deeplab/: ResNet-101 conv5 dilated + 1x1 classifier, bilinear
    upsampling to the input size); the deformable variant additionally uses deformable conv5."""
    def __init__(s, dcn, ncls=19):
        super().__init__(); s.body, c = resnet101_backbone(dcn); s.cls = nn.Conv2d(c, ncls, 1)
    def forward(s, x):
        return F.interpolate(s.cls(s.body(x)), size=x.shape[-2:], mode='bilinear', align_corners=False)

class RPNHead(nn.Module):
    def __init__(s, c, na=9):
        super().__init__(); s.conv = nn.Conv2d(c, 512, 3, padding=1); s.cls = nn.Conv2d(512, na, 1); s.box = nn.Conv2d(512, na * 4, 1)
    def forward(s, f): h = F.relu(s.conv(f)); return s.cls(h), s.box(h)

class RFCN(nn.Module):
    """R-FCN: ResNet-101 conv5 (dilated) -> 1x1 reduce -> position-sensitive score maps (k*k*(C+1)) and
    bbox maps (k*k*4) -> PS-RoI pooling over fixed RoIs -> vote (average over the k*k bins). The deformable
    variant uses deformable conv5 and deformable PS-RoI pooling (an offset applied to the RoI bins,
    predicted from the pooled features). Proposals come from an RPN on conv4-equivalent features here
    the RPN head output is part of the compared output and the RoIs for pooling are fixed."""
    def __init__(s, dcn, ncls=21, k=7):
        super().__init__(); s.body, c = resnet101_backbone(dcn); s.k = k; s.ncls = ncls; s.dcn = dcn
        s.reduce = nn.Conv2d(c, 1024, 1); s.rpn = RPNHead(c)
        s.cls_map = nn.Conv2d(1024, k * k * ncls, 1); s.box_map = nn.Conv2d(1024, k * k * 4, 1)
        s.register_buffer('rois', torch.tensor([[0, 0., 0., 3., 3.], [0, 1., 0., 3., 2.], [0, 0., 1., 2., 3.]]))   # on the /16 map of a 64x64 image (4x4)
        if dcn: s.off = nn.Conv2d(4, 2, 3, padding=1)   # per-bin (dy, dx) offsets from the first pooling, for the deformable PS-RoI pooling
    def forward(s, x):
        f = s.body(x); h = F.relu(s.reduce(f))
        rpn_cls, rpn_box = s.rpn(f)
        cm, bm = s.cls_map(h), s.box_map(h)
        if s.dcn:
            # deformable PS-RoI pooling (paper sec. 2.3): a first PS-RoI pooling of the bbox maps gives per-bin
            # features, a 1x1 conv (fc in the paper) predicts one (dy, dx) per bin, and the score / bbox maps
            # are pooled with the bins shifted by gamma=0.1 * (roi size) * offset, bilinearly sampled
            p0 = ps_roi_pool_ref(bm, s.rois, s.k)                                            # (R, 4, k, k)
            off = s.off(p0)                                                                   # (R, 2, k, k)
            cls = deform_ps_roi_pool_ref(cm, s.rois, s.k, off).mean(dim=(2, 3))
            box = deform_ps_roi_pool_ref(bm, s.rois, s.k, off).mean(dim=(2, 3))
            return torch.cat([rpn_cls.flatten(1), rpn_box.flatten(1), cls.flatten().unsqueeze(0), box.flatten().unsqueeze(0)], 1)
        cls = ps_roi_pool_ref(cm, s.rois, s.k).mean(dim=(2, 3))          # (R, ncls) votes
        box = ps_roi_pool_ref(bm, s.rois, s.k).mean(dim=(2, 3))          # (R, 4)
        return torch.cat([rpn_cls.flatten(1), rpn_box.flatten(1), cls.flatten().unsqueeze(0), box.flatten().unsqueeze(0)], 1)

class FasterRCNN(nn.Module):
    """Faster R-CNN (2fc head): ResNet-101 conv5 dilated -> RPN head; RoI features by PS-RoI-style
    average pooling of the reduced map over fixed RoIs (RoIPooling in the repo) -> 2 fc -> cls / box.
    The deformable variant: deformable conv5 + deformable RoI pooling (RoI shifted by predicted offsets)."""
    def __init__(s, dcn, ncls=21, k=7):
        super().__init__(); s.body, c = resnet101_backbone(dcn); s.k = k; s.dcn = dcn
        s.reduce = nn.Conv2d(c, 256, 1); s.rpn = RPNHead(c)
        s.fc1 = nn.Linear(256 * k * k, 1024); s.fc2 = nn.Linear(1024, 1024); s.cls = nn.Linear(1024, ncls); s.box = nn.Linear(1024, 4 * ncls)
        s.register_buffer('rois', torch.tensor([[0, 0., 0., 3., 3.], [0, 1., 0., 3., 2.], [0, 0., 1., 2., 3.]]))
        if dcn: s.off = nn.Linear(256 * k * k, 2 * k * k)   # per-bin (dy, dx) offsets (fc, as in the paper)
    def roi_pool(s, h, rois):   # average RoI pooling as PS-RoI pooling with C*k*k = the same map repeated per bin
        hk = h.repeat(1, s.k * s.k, 1, 1)                                 # (1, 256*k*k, H, W): group (i,j) = the map itself
        return ps_roi_pool_ref(hk, rois, s.k)                             # (R, 256, k, k)
    def forward(s, x):
        f = s.body(x); h = F.relu(s.reduce(f)); rpn_cls, rpn_box = s.rpn(f)
        rois = s.rois
        if s.dcn:   # deformable RoI pooling: offsets per bin predicted from a first (regular) pooling
            p0 = s.roi_pool(h, rois).flatten(1)
            off = s.off(p0).reshape(-1, 2, s.k, s.k)
            p = deform_ps_roi_pool_ref(h.repeat(1, s.k * s.k, 1, 1), rois, s.k, off).flatten(1)
            g = F.relu(s.fc2(F.relu(s.fc1(p))))
            return torch.cat([rpn_cls.flatten(1), rpn_box.flatten(1), s.cls(g).flatten().unsqueeze(0), s.box(g).flatten().unsqueeze(0)], 1)
        p = s.roi_pool(h, rois).flatten(1)
        g = F.relu(s.fc2(F.relu(s.fc1(p))))
        return torch.cat([rpn_cls.flatten(1), rpn_box.flatten(1), s.cls(g).flatten().unsqueeze(0), s.box(g).flatten().unsqueeze(0)], 1)

class FPN(nn.Module):
    """FPN on ResNet-101 (C2-C5 lateral 1x1 + top-down + 3x3 smooth, P6 by stride-2 pool) with the shared
    RPN head on every level; the deformable variant uses deformable 3x3 convs in conv5 (and, as in the
    repo's fpn/, the deformable RoI pooling is not exercised here: the compared output is the FPN + RPN
    head part, the RoI heads sit behind proposal selection)."""
    def __init__(s, dcn):
        super().__init__()
        r = M.resnet101(weights=None)
        if dcn:
            for blk in r.layer4:
                c = blk.conv2; blk.conv2 = DeformConv(c.in_channels, c.out_channels, c.stride[0], c.dilation[0])
        s.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool); s.c2, s.c3, s.c4, s.c5 = r.layer1, r.layer2, r.layer3, r.layer4
        s.lat = nn.ModuleList([nn.Conv2d(c, 256, 1) for c in (256, 512, 1024, 2048)])
        s.smooth = nn.ModuleList([nn.Conv2d(256, 256, 3, padding=1) for _ in range(4)])
        s.rpn = RPNHead(256, na=3)
    def forward(s, x):
        c2 = s.c2(s.stem(x)); c3 = s.c3(c2); c4 = s.c4(c3); c5 = s.c5(c4)
        p5 = s.lat[3](c5); p4 = s.lat[2](c4) + F.interpolate(p5, scale_factor=2, mode='nearest')
        p3 = s.lat[1](c3) + F.interpolate(p4, scale_factor=2, mode='nearest'); p2 = s.lat[0](c2) + F.interpolate(p3, scale_factor=2, mode='nearest')
        ps = [s.smooth[i](p) for i, p in enumerate((p2, p3, p4, p5))] + [F.max_pool2d(s.smooth[3](p5), 1, stride=2)]
        outs = []
        for p in ps: c, b = s.rpn(p); outs += [c.flatten(1), b.flatten(1)]
        return torch.cat(outs, 1)

class Op(nn.Module):   # the two operators on their own (the repo's deform_conv / deform_psroi demos)
    def __init__(s, which):
        super().__init__(); s.which = which
        if which == 'deform_conv': s.dc = DeformConv(64, 64, v2=False)
        elif which == 'deform_conv_v2': s.dc = DeformConv(64, 64, v2=True)
        else: s.register_buffer('rois', torch.tensor([[0, 0., 0., 11., 11.], [0, 4., 4., 15., 15.], [0, 2., 1., 7., 13.]])); s.off = nn.Conv2d(4, 2, 3, padding=1)
    def forward(s, x):
        if s.which.startswith('deform_conv'): return s.dc(x)
        if s.which == 'psroi_pool': return ps_roi_pool_ref(x, s.rois, 7)
        p0 = ps_roi_pool_ref(x[:, :49 * 4], s.rois, 7)                      # deform_psroi: per-bin offsets from a first pooling
        return deform_ps_roi_pool_ref(x, s.rois, 7, s.off(p0))

MODELS = {   # name -> (constructor, input)
    'deeplab':                  (lambda: DeepLab(False),   (1, 3, 64, 64)),
    'deeplab_dcn':              (lambda: DeepLab(True),    (1, 3, 64, 64)),
    'deeplab_voc':              (lambda: DeepLab(False, 21), (1, 3, 64, 64)),
    'deeplab_dcn_voc':          (lambda: DeepLab(True, 21),  (1, 3, 64, 64)),
    'rfcn':                     (lambda: RFCN(False),      (1, 3, 64, 64)),
    'rfcn_dcn':                 (lambda: RFCN(True),       (1, 3, 64, 64)),
    'rfcn_voc':                 (lambda: RFCN(False, 21),  (1, 3, 64, 64)),
    'rfcn_dcn_voc':             (lambda: RFCN(True, 21),   (1, 3, 64, 64)),
    'rcnn':                     (lambda: FasterRCNN(False),(1, 3, 64, 64)),
    'rcnn_dcn':                 (lambda: FasterRCNN(True), (1, 3, 64, 64)),
    'fpn':                      (lambda: FPN(False),       (1, 3, 64, 64)),
    'fpn_dcn':                  (lambda: FPN(True),        (1, 3, 64, 64)),
    'deform_conv':              (lambda: Op('deform_conv'),    (1, 64, 16, 16)),
    'deform_conv_v2':           (lambda: Op('deform_conv_v2'), (1, 64, 16, 16)),
    'psroi_pool':               (lambda: Op('psroi_pool'),     (1, 49 * 4, 16, 16)),
    'deform_psroi':             (lambda: Op('deform_psroi'),   (1, 49 * 4, 16, 16)),
}
MODELS['rfcn_coco'] = (lambda: RFCN(False, 81), (1, 3, 64, 64)); MODELS['rfcn_dcn_coco'] = (lambda: RFCN(True, 81), (1, 3, 64, 64))
MODELS['rcnn_coco'] = (lambda: FasterRCNN(False, 81), (1, 3, 64, 64)); MODELS['rcnn_dcn_coco'] = (lambda: FasterRCNN(True, 81), (1, 3, 64, 64))

if __name__ == '__main__':
    if sys.argv[1] == '--list': print('\n'.join(MODELS)); sys.exit(0)
    name, out, check = sys.argv[1:4]
    torch.manual_seed(0)
    ctor, shape = MODELS[name]; m = ctor()
    # BatchNorm: random affine, and running stats calibrated by one train-mode forward on a random batch
    # (momentum 1) -- with 100+ random BN layers, random running stats blow the ResNet-101 output up to
    # ~1e6 and the comparison degenerates into a relative-error test at float precision
    has_bn = False
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d): has_bn = True; mod.momentum = 1.0; mod.weight.data.uniform_(0.8, 1.2); mod.bias.data.normal_(0, 0.1)
    if has_bn:   # batch 1 (the RoI ops assume N=1); train-mode BN on a 64x64 map has enough samples
        m.train()
        with torch.no_grad(): m(torch.randn(*shape))
    m.eval(); x = torch.randn(*shape)
    with torch.no_grad(): ref = m(x)
    mod = fx.export_and_import(m, x, output_type='linalg-on-tensors', func_name='net')
    with open(out, 'wb') as f: mod.operation.print(file=f, binary=True)
    with open(check, 'wb') as f:
        f.write(struct.pack('i', 1)); f.write(struct.pack('i', x.numel())); f.write(x.numpy().astype(np.float32).tobytes())
        f.write(struct.pack('i', ref.numel())); f.write(np.ascontiguousarray(ref).astype(np.float32).tobytes())
    print('exported', name, 'input', list(x.shape), 'output', list(ref.shape), 'params %.1fM' % (sum(p.numel() for p in m.parameters()) / 1e6))
