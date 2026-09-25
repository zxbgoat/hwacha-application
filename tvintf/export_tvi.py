#!/usr/bin/env python3
"""Export one torchvision.transforms.v2 image transform (docs.pytorch.org/vision/stable/transforms.html)
through torch-mlir to linalg-on-tensors, plus a PyTorch reference for one call. Every case is a
single-input module net(x) on a 3x16x16 float image in [0, 1] (batched 1x3x16x16 where the transform
wants it) returning one float tensor. The Random* classes draw their
parameters and their apply-or-not coin inside forward (data dependent for torch.export even with p=1
and a degenerate range), so every random transform is pinned to the functional op the class applies
with one fixed draw (RandomRotation(30) -> F.rotate(x, 30), RandomCrop(8) -> F.crop at a fixed
offset). The reference is the genuine
transform / functional call; the exported body is the same call unless noted (rcase).
usage: export_tvi.py <name> <mlir_out> <check_out>"""
import sys, os, struct, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as NF
import torchvision.transforms.v2 as T, torchvision.transforms.v2.functional as F
from torchvision.transforms import InterpolationMode
from torch_mlir import fx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from intf_ops import add_ops

class Case(nn.Module):
    def __init__(s, body, **bufs):
        super().__init__(); s.body = body
        for k, v in bufs.items(): s.register_buffer(k, v)
    def forward(s, x): return s.body(s, x)
class Ref(Case):
    def __init__(s, body, ref, **bufs): super().__init__(body, **bufs); s.ref = ref
    def reference(s, x): return s.ref(s, x)

def rgb_to_hsv(x):
    """torchvision's _rgb_to_hsv written without in-place ops and aminmax (the export of adjust_hue fails
    inside torch-mlir's lowering: 'arith.cmpi' operands of different types)"""
    r, g, b = x[0], x[1], x[2]
    maxc = torch.maximum(torch.maximum(r, g), b); minc = torch.minimum(torch.minimum(r, g), b)
    eqc = maxc == minc; rng = maxc - minc; ones = torch.ones_like(maxc)
    sat = rng / torch.where(eqc, ones, maxc); div = torch.where(eqc, ones, rng)
    rc, gc, bc = (maxc - r) / div, (maxc - g) / div, (maxc - b) / div
    neq_r = maxc != r; eq_g = maxc == g
    h = (bc - gc) * (~neq_r).float() + (rc + 2.0 - bc) * (eq_g & neq_r).float() + (gc + 4.0 - rc) * (neq_r & ~eq_g).float()
    h = torch.remainder(h / 6.0 + 1.0, 1.0)
    return h, sat, maxc
def hsv_to_rgb(h, sat, v):
    h6 = h * 6; i = torch.floor(h6); f = h6 - i; i = torch.remainder(i, 6)
    q = ((1.0 - sat * f) * v).clamp(0, 1); t = ((sat * f + 1.0 - sat) * v).clamp(0, 1); p = ((1.0 - sat) * v).clamp(0, 1)
    sel = lambda k: (i == k).float()
    r = v * (sel(0) + sel(5)) + q * sel(1) + p * (sel(2) + sel(3)) + t * sel(4)
    g = t * sel(0) + v * (sel(1) + sel(2)) + q * sel(3) + p * (sel(4) + sel(5))
    b = p * (sel(0) + sel(1)) + t * sel(2) + v * (sel(3) + sel(4)) + q * sel(5)
    return torch.stack([r, g, b])
def adjust_hue(x, hf):
    h, sat, v = rgb_to_hsv(x); return hsv_to_rgb(torch.remainder(h + hf, 1.0), sat, v)
def adjust_saturation(x, f): return (f * x + (1 - f) * F.rgb_to_grayscale(x, 3)).clamp(0, 1)
def adjust_contrast(x, f): return (f * x + (1 - f) * F.rgb_to_grayscale(x).mean()).clamp(0, 1)
def adjust_brightness(x, f): return (x * f).clamp(0, 1)
def autocontrast(x):
    mn = x.amin(dim=(-2, -1), keepdim=True); mx = x.amax(dim=(-2, -1), keepdim=True); eq = mx == mn
    return ((x - torch.where(eq, torch.zeros_like(mn), mn)) / torch.where(eq, torch.ones_like(mn), mx - mn)).clamp(0, 1)
def equalize(x, levels):
    """torchvision's equalize on the uint8 image (x * 255): per channel a 256-bin histogram by one-hot
    counting, the cumulative histogram by a triangular matmul, the LUT with the step rule, applied by
    one-hot lookup. Integer divisions are floor(a / b) on exact integers held in float32."""
    v = torch.floor(x * 255)                                         # the uint8 values (the cast truncates)
    oh = (v[:, :, :, None] == levels[None, None, None, :]).float()   # (3, H, W, 256)
    hist = oh.sum(dim=(1, 2))                                        # (3, 256)
    cum = hist @ torch.triu(torch.ones(256, 256))                    # cum[k] = sum_{j <= k} hist[j]
    n = x.shape[1] * x.shape[2]
    last = torch.floor(((cum == n).float() * hist).amax(dim=-1, keepdim=True))   # hist at the first bin where cum == n (the last non-empty bin)
    step = torch.floor((n - last) / 255)
    lut = torch.floor((cum[:, :-1] + torch.floor(step / 2)) / torch.clamp(step, min=1)).clamp(0, 255)
    lut = torch.cat([torch.zeros(3, 1), lut], dim=-1)                # lut[0] = 0, lut[k] from cum[k - 1]
    out = (oh * lut[:, None, None, :]).sum(-1)
    return torch.where(step[:, :, None] > 0, out, v) / 255

def build(name):
    torch.manual_seed(0)
    C = {}
    def case(n, body, x, **b): C[n] = lambda: (Case(body, **b), x)
    def rcase(n, body, ref, x, **b): C[n] = lambda: (Ref(body, ref, **b), x)
    img = torch.rand(3, 16, 16); big = torch.rand(3, 24, 24); bimg = img[None]
    NN, BI = InterpolationMode.NEAREST, InterpolationMode.BILINEAR
    # ---- geometry
    case('resize', lambda s, x: T.Resize(8, antialias=False)(x), img)
    case('resize_bilinear', lambda s, x: T.Resize((24, 24), interpolation=BI, antialias=False)(x), img)
    case('center_crop', lambda s, x: T.CenterCrop(8)(x), img)
    case('random_crop', lambda s, x: F.crop(x, 3, 5, 8, 8), img)                                   # RandomCrop(8): one draw (top 3, left 5)
    case('random_resized_crop', lambda s, x: F.resized_crop(x, 2, 3, 10, 12, [16, 16], interpolation=BI, antialias=False), img)
    case('random_horizontal_flip', lambda s, x: F.horizontal_flip(x), img)
    case('random_vertical_flip', lambda s, x: F.vertical_flip(x), img)
    case('pad', lambda s, x: T.Pad(2)(x), img)
    case('pad_reflect', lambda s, x: T.Pad(3, padding_mode='reflect')(x), img)
    case('random_zoom_out', lambda s, x: F.pad(x, [4, 2, 3, 5], fill=0.5), img)                    # RandomZoomOut: one draw of the padding
    case('random_rotation', lambda s, x: F.rotate(x, 30.0, interpolation=BI), img)
    case('random_rotation_nearest', lambda s, x: F.rotate(x, 45.0, interpolation=NN, expand=False), img)
    case('random_affine', lambda s, x: F.affine(x, angle=15.0, translate=[0, 0], scale=1.2, shear=[10.0, 0.0], interpolation=BI), img)
    pc = F._geometry._get_perspective_coeffs([[0, 0], [15, 0], [0, 15], [15, 15]], [[1, 2], [13, 1], [2, 14], [14, 12]])   # the lstsq of the 8 coefficients, solved on the host
    case('random_perspective', lambda s, x: F.perspective(x, None, None, interpolation=BI, coefficients=pc), img)
    case('elastic_transform', lambda s, x: F.elastic_transform(x, s.d, interpolation=BI), img, d=torch.randn(1, 16, 16, 2) * 0.05)   # ElasticTransform: one displacement draw
    case('five_crop', lambda s, x: torch.stack(T.FiveCrop(8)(x)), img)
    case('ten_crop', lambda s, x: torch.stack(T.TenCrop(8)(x)), img)
    case('random_iou_crop', lambda s, x: F.crop(x, 4, 2, 10, 12), img)                            # RandomIoUCrop: the crop of one accepted draw
    case('random_resize', lambda s, x: F.resize(x, [12, 12], antialias=False), img)
    case('random_shortest_size', lambda s, x: F.resize(x, [12, 12], antialias=False), big)
    case('scale_jitter', lambda s, x: F.resize(x, [20, 20], antialias=False), img)               # ScaleJitter: one scale draw (1.25)
    # ---- color
    rcase('color_jitter', lambda s, x: adjust_hue(adjust_saturation(adjust_contrast(adjust_brightness(x, 1.3), 0.8), 1.4), 0.1),
          lambda s, x: F.adjust_hue(F.adjust_saturation(F.adjust_contrast(F.adjust_brightness(x, 1.3), 0.8), 1.4), 0.1), img)   # one draw, the class' order
    case('grayscale', lambda s, x: T.Grayscale()(x), img)
    case('grayscale_3ch', lambda s, x: T.Grayscale(num_output_channels=3)(x), img)
    case('random_grayscale', lambda s, x: F.rgb_to_grayscale(x, num_output_channels=3), img)
    case('rgb', lambda s, x: T.RGB()(x), img[:1])
    case('random_channel_permutation', lambda s, x: F.permute_channels(x, [2, 0, 1]), img)
    rcase('random_photometric_distort', lambda s, x: adjust_hue(adjust_saturation(adjust_contrast(adjust_brightness(x, 0.9), 1.3), 0.7), -0.05),
          lambda s, x: F.adjust_hue(F.adjust_saturation(F.adjust_contrast(F.adjust_brightness(x, 0.9), 1.3), 0.7), -0.05), img)
    case('random_adjust_sharpness', lambda s, x: F.adjust_sharpness(x, 2.0), img)
    rcase('random_autocontrast', lambda s, x: autocontrast(x), lambda s, x: F.autocontrast(x), img)
    rcase('random_equalize', lambda s, x: equalize(x, s.levels), lambda s, x: F.equalize((x * 255).to(torch.uint8)).float() / 255, img, levels=torch.arange(256).float())
    case('random_invert', lambda s, x: F.invert(x), img)
    case('random_posterize', lambda s, x: F.posterize((x * 255).to(torch.uint8), 3).float() / 255, img)
    case('random_solarize', lambda s, x: F.solarize(x, 0.5), img)
    case('gaussian_blur', lambda s, x: F.gaussian_blur(x, [5, 5], sigma=[1.5, 1.5]), img)
    case('gaussian_noise', lambda s, x: x + 0.1 * s.noise, img, noise=torch.randn(3, 16, 16))          # GaussianNoise(sigma=0.1): one noise draw
    case('random_erasing', lambda s, x: F.erase(x, 3, 4, 6, 7, s.v), img, v=torch.tensor(0.0))   # RandomErasing: one box draw
    case('normalize', lambda s, x: T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(x), img)
    case('linear_transformation', lambda s, x: T.LinearTransformation(s.M, s.mv)(x), img, M=torch.randn(768, 768) * 0.05, mv=torch.rand(768))
    # ---- batch / video
    lab = torch.tensor([1, 3])
    case('cutmix', lambda s, x: x * s.mask + x.flip(0) * (1 - s.mask), torch.rand(2, 3, 16, 16), mask=(lambda m: (m.__setitem__((slice(None), slice(None), slice(4, 12), slice(2, 10)), 0), m)[1])(torch.ones(1, 1, 16, 16)))   # CutMix: one box draw (label mixing not exported)
    case('mixup', lambda s, x: 0.7 * x + 0.3 * x.flip(0), torch.rand(2, 3, 16, 16))               # MixUp: lambda = 0.7
    case('uniform_temporal_subsample', lambda s, x: T.UniformTemporalSubsample(4)(x), torch.rand(8, 3, 8, 8))
    # ---- torchvision.ops (docs.pytorch.org/vision/stable/ops.html): intf_ops.py
    add_ops(case, rcase, lambda *sh: torch.randn(*sh))
    if name == '--list': return sorted(C)
    if name not in C: raise SystemExit('unknown case ' + name)
    return C[name]()

if __name__ == '__main__':
    if sys.argv[1] == '--list': print('\n'.join(build('--list'))); sys.exit(0)
    name, mlir_out, bin_out = sys.argv[1:4]
    m, x = build(name); m = m.eval()
    with torch.no_grad(): y = m.reference(x) if hasattr(m, 'reference') else m(x)
    with torch.no_grad(): assert torch.allclose(m(x), y, atol=1e-4, rtol=1e-4), 'exported body != reference'
    print('%s: in %s -> out %s' % (name, list(x.shape), list(y.shape)))
    mod = fx.export_and_import(m, x, output_type='linalg-on-tensors', func_name='net')
    open(mlir_out, 'w').write(str(mod))
    xf = np.ascontiguousarray(x.numpy()).astype(np.float32).ravel(); yf = np.ascontiguousarray(y.numpy()).astype(np.float32).ravel()
    with open(bin_out, 'wb') as f:
        f.write(struct.pack('i', xf.size)); f.write(xf.tobytes()); f.write(struct.pack('i', yf.size)); f.write(yf.tobytes())
