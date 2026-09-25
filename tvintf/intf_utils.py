"""torchvision.utils cases (docs.pytorch.org/vision/stable/utils.html) for export_tvi.py. make_grid is
tensor code; save_image is make_grid plus the uint8 quantisation that goes into the PNG (the reference
reads the written file back); the draw_* functions rasterise with PIL and flow_to_image indexes a
colour wheel, so their exported graphs are tensor re-implementations of the same rasterisation on
coordinate grids (rectangle outlines of a given width, filled circles, mask overlays with the
overlap rule, one-hot colour-wheel lookup with atan2 from atan); the reference is the genuine call."""
import io, math, torch, torch.nn.functional as NF, torchvision.utils as U
from torchvision.utils import _make_colorwheel

def rect_outline(ys, xs, box, w):
    """pixels of PIL's rectangle outline of width w for the inclusive box [x1, y1, x2, y2]"""
    x1, y1, x2, y2 = box
    outer = (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)
    inner = (xs >= x1 + w) & (xs <= x2 - w) & (ys >= y1 + w) & (ys <= y2 - w)
    return outer & ~inner
def draw_boxes(img, ys, xs, boxes, colors, w):
    """img (3, H, W) float 0..255; boxes a python list of [x1, y1, x2, y2] (constants); later boxes paint over earlier ones"""
    out = img
    for k in range(len(boxes)):
        m = rect_outline(ys, xs, boxes[k], w).float()
        out = out * (1 - m) + colors[k][:, None, None] * m
    return out
def draw_masks(img, masks, colors, alpha):
    """torchvision's draw_segmentation_masks: each mask paints its colour, overlapping pixels 0, then blend"""
    draw = img
    for k in range(masks.shape[0]):
        m = masks[k]; draw = draw * (1 - m) + colors[k][:, None, None] * m
    over = (masks.sum(0) > 1).float()
    draw = draw * (1 - over)
    return torch.floor(img * (1 - alpha) + draw * alpha)
def draw_points(img, ys, xs, pts, color, r):
    """filled circles of radius r (PIL's ellipse over the bbox [x - r, y - r, x + r, y + r]) at the keypoints"""
    out = img
    for k in range(len(pts)):
        cx, cy = pts[k]
        m = ((xs - cx) ** 2 + (ys - cy) ** 2 <= (r + 0.5) ** 2).float()   # PIL fills the ellipse of the (2r + 1)-pixel bbox: radius r + 0.5 about the centre
        out = out * (1 - m) + color[:, None, None] * m
    return out
def atan2(y, x):
    """atan2 from atan: quadrant fix-ups; x == 0 handled by the sign of y (the flow here has no zero x)"""
    ax = torch.where(x == 0, torch.full_like(x, 1e-30), x)
    a = torch.atan(y / ax)
    return torch.where(x < 0, a + torch.where(y >= 0, torch.full_like(a, math.pi), torch.full_like(a, -math.pi)), a)
def flow_to_image(flow, wheel, ks):
    """torchvision's flow_to_image: normalise by the max norm, angle -> colour-wheel position, linear
    interpolation between the two neighbouring wheel entries (one-hot lookup), fade to white by the norm"""
    norm0 = torch.sqrt((flow ** 2).sum(1))
    nf = flow / (norm0.max() + 1e-5)
    norm = torch.sqrt((nf ** 2).sum(1))
    a = atan2(-nf[:, 1], -nf[:, 0]) / math.pi
    fk = (a + 1) / 2 * (wheel.shape[0] - 1)
    k0 = torch.floor(fk); k1 = torch.where(k0 + 1 == wheel.shape[0], torch.zeros_like(k0), k0 + 1); f = fk - k0
    oh0 = (k0[..., None] == ks).float(); oh1 = (k1[..., None] == ks).float()      # (N, H, W, 55)
    col0 = oh0 @ (wheel / 255.0); col1 = oh1 @ (wheel / 255.0)                       # (N, H, W, 3)
    col = (1 - f[..., None]) * col0 + f[..., None] * col1
    col = 1 - norm[..., None] * (1 - col)
    return torch.floor(255 * col).permute(0, 3, 1, 2)

def add_utils(case, rcase, R):
    torch.manual_seed(4)
    imgs = torch.rand(4, 3, 8, 8)
    case('make_grid', lambda s, x: U.make_grid(x, nrow=2, padding=1, pad_value=0.5), imgs)
    # (normalize=True clamps with the tensor's own min / max as python floats: data dependent for the
    # export; the same per-image normalisation written with tensor ops)
    def norm_each(x):
        mn = x.amin(dim=(1, 2, 3), keepdim=True); mx = x.amax(dim=(1, 2, 3), keepdim=True); return (x - mn) / (mx - mn + 1e-5)
    rcase('make_grid_normalize', lambda s, x: U.make_grid(norm_each(x), nrow=2, padding=1), lambda s, x: U.make_grid(x, nrow=2, padding=1, normalize=True, scale_each=True), imgs * 3 - 1)
    def saved_png(x):
        buf = io.BytesIO(); U.save_image(x, buf, format='png', nrow=2, padding=1); buf.seek(0)
        from PIL import Image; import numpy as np
        return torch.from_numpy(np.array(Image.open(buf))).permute(2, 0, 1).float()
    rcase('save_image', lambda s, x: torch.floor((U.make_grid(x, nrow=2, padding=1) * 255 + 0.5).clamp(0, 255)), lambda s, x: saved_png(x), imgs)
    H = W = 16; ys, xs = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing='ij')
    img8 = (torch.rand(3, H, W) * 255).to(torch.uint8); imgf = img8.float()
    boxes = torch.tensor([[1., 2., 9., 10.], [6., 5., 14., 13.], [0., 12., 5., 15.]]); cols = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]; colt = torch.tensor(cols).float()
    bl = boxes.tolist()
    rcase('draw_bounding_boxes', lambda s, x: draw_boxes(x, s.ys, s.xs, bl, s.cols, 2), lambda s, x: U.draw_bounding_boxes(x.to(torch.uint8), s.boxes, colors=cols, width=2).float(), imgf, ys=ys, xs=xs, boxes=boxes, cols=colt)
    masks = torch.zeros(2, H, W, dtype=torch.bool); masks[0, 2:9, 3:12] = True; masks[1, 6:14, 8:15] = True
    rcase('draw_segmentation_masks', lambda s, x: draw_masks(x, s.masks, s.cols, 0.6), lambda s, x: U.draw_segmentation_masks(x.to(torch.uint8), s.masks_b, alpha=0.6, colors=cols[:2]).float(), imgf, masks=masks.float(), masks_b=masks, cols=colt[:2])
    kps = torch.tensor([[[3., 4.], [10., 6.], [7., 12.]]])
    kl = kps[0].tolist()
    rcase('draw_keypoints', lambda s, x: draw_points(x, s.ys, s.xs, kl, s.col, 2), lambda s, x: U.draw_keypoints(x.to(torch.uint8), s.kps, colors=(255, 0, 0), radius=2).float(), imgf, ys=ys, xs=xs, kps=kps, col=colt[0])
    wheel = _make_colorwheel().float(); flow = torch.randn(1, 2, 8, 8)
    rcase('flow_to_image', lambda s, x: flow_to_image(x, s.wheel, s.ks), lambda s, x: U.flow_to_image(x).float(), flow, wheel=wheel, ks=torch.arange(wheel.shape[0]).float())
