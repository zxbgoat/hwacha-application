#!/usr/bin/env python3
"""Export one pytorchvideo model-zoo network (pytorchvideo.readthedocs.io/en/latest/model_zoo.html, the
TorchHub builders of pytorchvideo.models.hub) to linalg-on-tensors MLIR via torch-mlir, with a PyTorch
reference.  usage: export_tvid.py <case> <out.mlir> <out_check.bin>   |   export_tvid.py --list

Every case is the hub builder's architecture at its full width (random weights, fixed seed) on a small
clip: T frames (the setting's frame length) at HWxHW instead of 224x224. Every head pool that is sized
for 224x224 (an AvgPool3d((T', 7, 7)): on the 7x7 map after /32 it is the global average) is replaced
by the global average it computes there, so the model takes the small clip. Models run in eval mode;
the case is a single-input module net(x) -> one float tensor."""
import sys, struct, numpy as np, torch, torch.nn as nn, torch.nn.functional as NF
from torch_mlir import fx
import pytorchvideo.models.hub as H
import pytorchvideo.models.head as PH
import inspect, torch_mlir.extras.fx_importer as _fxi
# torch-mlir turns every constant tensor into a literal via np.array(tensor.tolist()), one Python float
# per element (~30x the tensor); convert through numpy instead (as torchvision/export_tv.py does).
def _tensor_to_numpy(tensor, npy_dtype):
    try:
        with torch.utils._mode_utils.no_dispatch():
            return np.asarray(tensor.detach().cpu().numpy()).astype(npy_dtype, copy=False)
    except Exception:
        return np.array(tensor.tolist()).astype(npy_dtype)
_src = inspect.getsource(_fxi._make_vtensor_literal_op)
assert "np.array(tensor.tolist()).astype(npy_dtype)" in _src
_fxi._tensor_to_numpy = _tensor_to_numpy
exec(_src.replace("np.array(tensor.tolist()).astype(npy_dtype)", "_tensor_to_numpy(tensor, npy_dtype)"), _fxi.__dict__)

HW = 32          # spatial size of the clip (32 -> 1x1 after the /32 trunk)

class Wrap(nn.Module):
    """Single-input module: the clip (or the (slow, fast) clips stacked along T) -> one float tensor."""
    def __init__(s, m, f): super().__init__(); s.m = m; s.f = f
    def forward(s, x): return s.f(s.m, x)

def global_heads(m):
    """Replace every head AvgPool3d (sized for the 224x224 clip: (T', 7, 7) on the 7x7 map after /32) by
    the global average it is there, so the trunk output of the small clip is pooled the same way. The RoI heads' (T, 1, 1) temporal pools stay."""
    for mod in m.modules():
        if isinstance(mod, PH.ResNetBasicHead) and isinstance(mod.pool, nn.AvgPool3d):
            mod.pool = nn.AdaptiveAvgPool3d(1)
        if isinstance(getattr(mod, 'pool', None), nn.ModuleList):     # SlowFast PoolConcatPathway: one pool per pathway
            mod.pool = nn.ModuleList([nn.AdaptiveAvgPool3d(1) if isinstance(p, nn.AvgPool3d) and p.kernel_size[1] > 1 else p for p in mod.pool])
        if type(mod).__name__ == 'ProjectedPool' and isinstance(getattr(mod, 'pool', None), nn.AvgPool3d):
            mod.pool = nn.AdaptiveAvgPool3d(1)   # X3D head: pre_conv -> pool -> post_conv
    return m

class DepthwiseConv3d(nn.Module):
    """nn.Conv3d with groups == channels (CSN, X3D): torch-mlir marks aten.convolution with groups on 5-D
    input illegal but lowers the 4-D depthwise convolution. Split the kernel along its kT temporal taps:
    each tap is a 2-D depthwise convolution over the (B*T) frames (T folded into the batch), the taps are
    summed with their temporal offsets on the zero-padded, temporally strided frame axis."""
    def __init__(s, c):
        super().__init__(); s.c = c
        assert c.groups == c.in_channels == c.out_channels and c.dilation == (1, 1, 1)
    def forward(s, x):
        c = s.c; B, C, T, H, W = x.shape; kT, kH, kW = c.kernel_size; sT, sH, sW = c.stride; pT, pH, pW = c.padding
        xt = NF.pad(x, (0, 0, 0, 0, pT, pT))                                   # (B, C, T + 2pT, H, W)
        To = (T + 2 * pT - kT) // sT + 1
        out = None
        for t in range(kT):
            fr = xt[:, :, t:t + sT * (To - 1) + 1:sT]                          # (B, C, To, H, W) frames of this tap
            fr = fr.permute(0, 2, 1, 3, 4).reshape(B * To, C, H, W)
            y = NF.conv2d(fr, c.weight[:, :, t], None, (sH, sW), (pH, pW), 1, C)
            out = y if out is None else out + y
        out = out.reshape(B, To, C, out.shape[-2], out.shape[-1]).permute(0, 2, 1, 3, 4)
        return out if c.bias is None else out + c.bias[None, :, None, None, None]

def depthwise_2d(m):
    """swap every depthwise Conv3d for its temporal-tap decomposition"""
    for name, mod in list(m.named_modules()):
        for cn, ch in list(mod.named_children()):
            if isinstance(ch, nn.Conv3d) and ch.groups > 1: setattr(mod, cn, DepthwiseConv3d(ch))
    return m

class RoIAlignGather(nn.Module):
    """torchvision.ops.RoIAlign (no torch-mlir lowering) for fixed boxes on a (1, C, H, W) map as one
    constant gather: every (bin, sample) point's bilinear weights on the map are precomputed into a
    (R*k*k, H*W) matrix, so the RoI pooling is a matmul over the flattened map (the bilinear sampling and
    border clamping of torchvision's kernel, sampling_ratio=0 -> ceil(roi / k) samples per bin, as in
    tvintf/intf_ops.py)."""
    def __init__(s, roi, boxes, H, W):
        super().__init__(); k = roi.output_size[0]; scale = roi.spatial_scale; off = 0.5 if roi.aligned else 0.0
        rows = []
        for r in range(boxes.shape[0]):
            x1, y1, x2, y2 = [float(v) * scale - off for v in boxes[r, 1:]]
            rw, rh = x2 - x1, y2 - y1
            if not roi.aligned: rw, rh = max(rw, 1.0), max(rh, 1.0)
            bw, bh = rw / k, rh / k
            srh = roi.sampling_ratio or int(np.ceil(rh / k)); srw = roi.sampling_ratio or int(np.ceil(rw / k))
            for i in range(k):
                for j in range(k):
                    m = np.zeros((H, W), np.float64)
                    for a in range(srh):
                        for b in range(srw):
                            py = y1 + i * bh + (a + 0.5) * bh / srh; px = x1 + j * bw + (b + 0.5) * bw / srw
                            if py < -1 or py > H or px < -1 or px > W: continue
                            py = max(py, 0.0); px = max(px, 0.0)
                            yl, xl = int(py), int(px)
                            if yl >= H - 1: yl = H - 1; py = float(yl)
                            if xl >= W - 1: xl = W - 1; px = float(xl)
                            yh, xh = min(yl + 1, H - 1), min(xl + 1, W - 1)
                            ly, lx = py - yl, px - xl; hy, hx = 1 - ly, 1 - lx
                            m[yl, xl] += hy * hx; m[yl, xh] += hy * lx; m[yh, xl] += ly * hx; m[yh, xh] += ly * lx
                    rows.append(m.reshape(-1) / (srh * srw))
        s.register_buffer('G', torch.tensor(np.stack(rows), dtype=torch.float32))   # (R*k*k, H*W)
        s.R, s.k = boxes.shape[0], k
    def forward(s, x, boxes):
        C = x.shape[1]
        v = x.reshape(C, -1) @ s.G.t()                                            # (C, R*k*k)
        return v.reshape(C, s.R, s.k, s.k).permute(1, 0, 2, 3)

def roi_gather(m, boxes, H, W):
    """swap the RoIAlign of every RoI head for the constant gather on the (H, W) map it sees"""
    for mod in m.modules():
        if isinstance(mod, PH.ResNetRoIHead): mod.roi_layer = RoIAlignGather(mod.roi_layer, boxes, H, W)
    m.register_buffer('boxes', boxes)
    return m

def mvit_const_pos(m):
    """MViT: torch.cat((cls_token.expand(B, -1, -1), x)) and the (spatial + temporal) positional embedding are
    computed from nn.Parameters inside the forward, which torch-mlir's importer rejects inside aten.cat's
    list argument ("heterogeneous list"). In eval both are constants: precompute the full positional
    embedding (cls row included) and the cls token as plain buffers."""
    import pytorchvideo.layers.positional_encoding as PE
    for mod in m.modules():
        if isinstance(mod, PE.SpatioTemporalClsPositionalEncoding):
            with torch.no_grad():
                if mod.sep_pos_embed:
                    pe = mod.pos_embed_spatial.repeat(1, mod.num_temporal_patch, 1) + torch.repeat_interleave(mod.pos_embed_temporal, mod.num_spatial_patch, dim=1)
                    if mod.cls_embed_on: pe = torch.cat([mod.pos_embed_class, pe], 1)
                else: pe = mod.pos_embed
                mod.register_buffer('pe_const', pe.clone())
                if mod.cls_embed_on: mod.register_buffer('cls_const', mod.cls_token.detach().clone())
            def fwd(mod):
                def f(x):
                    if mod.cls_embed_on: x = torch.cat((mod.cls_const.expand(x.shape[0], -1, -1), x), 1)
                    return x + mod.pe_const
                return f
            mod.forward = fwd(mod)
    return m

def split_sf(x, alpha):
    """(1, 3, T + alpha*T, H, W) -> [slow (1,3,T,H,W), fast (1,3,alpha*T,H,W)]"""
    T = x.shape[2] // (alpha + 1)
    return [x[:, :, :T], x[:, :, T:]]

def build(name):
    torch.manual_seed(0)
    C = {}
    def case(n, mk, T, f=lambda m, x: m(x), hw=HW, T_in=None):
        C[n] = lambda: (mk, T, f, hw, T_in)
    # ---- Kinetics-400 ResNets (frame length x sample rate): the clip length is the setting's
    case('c2d_r50', lambda: H.c2d_r50(), 8)
    case('i3d_r50', lambda: H.i3d_r50(), 8)
    case('slow_r50', lambda: H.slow_r50(), 8)
    case('slow_r50_4x16', lambda: H.slow_r50(), 4)
    case('r2plus1d_r50', lambda: H.r2plus1d_r50(), 16)
    case('csn_r101', lambda: H.csn_r101(), 32)
    # ---- SlowFast: input is [slow, fast] with fast = alpha x slow frames; one tensor holds both along T
    for n, mk, T, a in (('slowfast_r50', H.slowfast_r50, 8, 4), ('slowfast_r50_4x16', H.slowfast_r50, 4, 4),
                        ('slowfast_r101', H.slowfast_r101, 8, 4), ('slowfast_16x8_r101_50_50', H.slowfast_16x8_r101_50_50, 16, 4)):
        case(n, (lambda mk: lambda: mk())(mk), T, (lambda a: lambda m, x: m(split_sf(x, a)))(a), T_in=T * (a + 1))
    # ---- X3D: the hub builders fix input_crop_size (160 / 224 / 312), from which create_x3d sizes the head
    #      pool; same architecture (hub clip length, width 2.0, depth 2.2 / 5.0 for L) built for the small clip
    from pytorchvideo.models.x3d import create_x3d
    case('x3d_xs', lambda: create_x3d(input_clip_length=4, input_crop_size=HW), 4)
    case('x3d_s', lambda: create_x3d(input_clip_length=13, input_crop_size=HW), 13)
    case('x3d_m', lambda: create_x3d(input_clip_length=16, input_crop_size=HW), 16)
    case('x3d_l', lambda: create_x3d(input_clip_length=16, input_crop_size=HW, depth_factor=5.0), 16)
    # ---- MViT: the patch embedding and pooling attention are sized by (spatial_size, temporal_size)
    case('mvit_base_16x4', lambda: mvit_const_pos(H.mvit_base_16x4(spatial_size=HW)), 16)
    case('mvit_base_32x3', lambda: mvit_const_pos(H.mvit_base_32x3(spatial_size=HW)), 32)
    # ---- accelerator zoo (mobile CPU): EfficientX3d XS / S
    case('efficient_x3d_xs', lambda: H.efficient_x3d_xs(), 4)
    case('efficient_x3d_s', lambda: H.efficient_x3d_s(), 13)
    # ---- AVA detection: the network on a 64x64 clip and 2 fixed boxes (N x 5: batch index, x1, y1, x2, y2), the
    #      RoIAlign on the /16 map as a constant gather; output (2, 80) sigmoid scores
    boxes = torch.tensor([[0., 8., 8., 40., 48.], [0., 20., 4., 60., 30.]])
    case('slow_r50_detection', lambda: roi_gather(H.slow_r50_detection(), boxes, 4, 4), 4, f=lambda m, x: m(x, m.boxes), hw=64)
    case('slowfast_r50_detection', lambda: roi_gather(H.slowfast_r50_detection(), boxes, 4, 4), 8, f=lambda m, x: m(split_sf(x, 4), m.boxes), hw=64, T_in=40)
    if name == '--list': return sorted(C)
    if name not in C: raise SystemExit('unknown case ' + name)
    return C[name]()

def prepare(name):
    mk, T, f, hw, T_in = build(name)
    m = depthwise_2d(global_heads(mk()))
    # untrained BN sits at its zero-mean/unit-var fixed point, which drives the logits to ~0; random
    # running stats keep the forward non-degenerate (the same weights feed the reference and the build)
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm3d, nn.BatchNorm2d)):
            mod.running_mean.normal_(0, 0.1); mod.running_var.uniform_(0.5, 1.5)
            mod.weight.data.uniform_(0.5, 1.5); mod.bias.data.normal_(0, 0.1)
    w = Wrap(m, f).eval()
    x = torch.randn(1, 3, T_in or T, hw, hw)
    return w, x

if __name__ == '__main__':
    if sys.argv[1] == '--list': print('\n'.join(build('--list'))); sys.exit(0)
    name, out, check = sys.argv[1:4]
    w, x = prepare(name)
    with torch.no_grad(): ref = w(x)
    mod = fx.export_and_import(w, x, output_type='linalg-on-tensors', func_name='net')
    with open(out, 'wb') as fo: mod.operation.print(file=fo, binary=True)
    with open(check, 'wb') as fo:
        fo.write(struct.pack('i', 1))
        fo.write(struct.pack('i', x.numel())); fo.write(x.numpy().astype(np.float32).tobytes())
        fo.write(struct.pack('i', ref.numel())); fo.write(np.ascontiguousarray(ref).astype(np.float32).tobytes())
    print('exported', name, 'input', list(x.shape), 'output', list(ref.shape), 'params %.1fM' % (sum(p.numel() for p in w.m.parameters()) / 1e6))
