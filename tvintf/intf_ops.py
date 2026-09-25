"""torchvision.ops cases (docs.pytorch.org/vision/stable/ops.html) for export_tvi.py. The box utilities,
losses and layers export as they are; the C++ operators (nms, roi_align, roi_pool, ps_roi_*,
deform_conv2d) and the index-returning utilities have no torch-mlir lowering, so their exported
graphs are fixed-size tensor compositions (the reference is the genuine torchvision call):
  nms / batched_nms   boxes ranked by score (rank permutation), then greedy suppression unrolled over
                      the 8 boxes; the result is the keep mask in input order (nms returns the kept
                      indices sorted by score: the same information)
  remove_small_boxes  the keep mask
  masks_to_boxes      min / max of the masked coordinate grids
  roi_align / MultiScaleRoIAlign   the CUDA kernel's sampling (sr x sr bilinear samples per bin, border
                      clamping) with one-hot gathers (torchvision's pure-tensor _roi_align fails to lower)
  ps_roi_align        the position-sensitive variant of the same sampling
  roi_pool            max over constant per-bin pixel masks (the RoIs are constants)
  ps_roi_pool, deform_conv2d       ../deformable/dcn_ops.py (gather + bilinear + matmul)"""
import os, sys, math, torch, torch.nn as nn, torch.nn.functional as NF, torchvision.ops as O
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'deformable'))
import dcn_ops

def iou_matrix(b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = torch.maximum(b[:, None, :2], b[None, :, :2]); rb = torch.minimum(b[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0); inter = wh[..., 0] * wh[..., 1]
    return inter / (area[:, None] + area[None, :] - inter)
def nms_mask(boxes, scores, thr, ar):
    """keep mask of greedy NMS: process boxes in decreasing score order, box i is kept iff no earlier
    kept box overlaps it above thr. P[i, j] = 1 iff box j has rank i (rank = # boxes with a larger score)."""
    n = boxes.shape[0]
    rank = (scores[None, :] > scores[:, None]).float().sum(-1)            # rank[j] = # boxes scoring higher than j
    P = (rank[None, :] == ar[:, None]).float()                            # (n, n), sorted = P @ x
    bs = P @ boxes; iou = iou_matrix(bs); over = (iou > thr).float()
    keep = []
    for i in range(n):
        sup = sum(keep[j] * over[i, j] for j in range(i)) if i else torch.zeros(())
        keep.append(1.0 - torch.clamp(sup, max=1.0))
    ks = torch.stack(keep)
    return P.t() @ ks                                                     # back to input order
def masks_to_boxes(m, ys, xs):
    """m (N, H, W) 0/1; boxes [xmin, ymin, xmax, ymax] over the masked pixels"""
    big = 1e6
    xmin = torch.amin(xs * m + big * (1 - m), dim=(1, 2)); ymin = torch.amin(ys * m + big * (1 - m), dim=(1, 2))
    xmax = torch.amax(xs * m - big * (1 - m), dim=(1, 2)); ymax = torch.amax(ys * m - big * (1 - m), dim=(1, 2))
    return torch.stack([xmin, ymin, xmax, ymax], 1)
def roi_pool_masks(rois, H, W, k, scale):
    """torchvision's roi_pool bin rule for constant rois -> (R, k, k, H, W) 0/1 masks"""
    R = rois.shape[0]; M = torch.zeros(R, k, k, H, W)
    for r in range(R):
        x1 = round(float(rois[r, 1]) * scale); y1 = round(float(rois[r, 2]) * scale)
        x2 = round(float(rois[r, 3]) * scale); y2 = round(float(rois[r, 4]) * scale)
        rw = max(x2 - x1 + 1, 1); rh = max(y2 - y1 + 1, 1); bw = rw / k; bh = rh / k
        for i in range(k):
            for j in range(k):
                hs = min(max(int(math.floor(i * bh)) + y1, 0), H); he = min(max(int(math.ceil((i + 1) * bh)) + y1, 0), H)
                ws = min(max(int(math.floor(j * bw)) + x1, 0), W); we = min(max(int(math.ceil((j + 1) * bw)) + x1, 0), W)
                if he > hs and we > ws: M[r, i, j, hs:he, ws:we] = 1
    return M
def ps_roi_align(x, rois, k, scale, sr):
    """x (1, C*k*k, H, W), rois (R, 5) -> (R, C, k, k); bin (i, j) of roi r averages sr x sr bilinear samples
    of channel c*k*k + i*k + j (torchvision's ps_roi_align, aligned=False semantics: roi_start = x1 * scale - 0.5)"""
    R = rois.shape[0]; CK = x.shape[1]; C = CK // (k * k)
    outs = []
    for r in range(R):
        x1 = rois[r, 1] * scale - 0.5; y1 = rois[r, 2] * scale - 0.5; x2 = rois[r, 3] * scale - 0.5; y2 = rois[r, 4] * scale - 0.5
        rw = torch.clamp(x2 - x1, min=0.1); rh = torch.clamp(y2 - y1, min=0.1); bw = rw / k; bh = rh / k
        bins = []
        for i in range(k):
            for j in range(k):
                py = torch.stack([y1 + i * bh + (a + 0.5) * bh / sr for a in range(sr) for b in range(sr)])
                px = torch.stack([x1 + j * bw + (b + 0.5) * bw / sr for a in range(sr) for b in range(sr)])
                s = dcn_ops.bilinear_gather(x, py[None], px[None]).mean(-1)[0]      # (C*k*k,)
                bins.append(s.reshape(C, k * k)[:, i * k + j])
        outs.append(torch.stack(bins, 1).reshape(C, k, k))
    return torch.stack(outs)

def inter_union(b1, b2, eps=1e-7):
    x1, y1, x2, y2 = b1.unbind(-1); x1g, y1g, x2g, y2g = b2.unbind(-1)
    xk1 = torch.maximum(x1, x1g); yk1 = torch.maximum(y1, y1g); xk2 = torch.minimum(x2, x2g); yk2 = torch.minimum(y2, y2g)
    inter = torch.where((yk2 > yk1) & (xk2 > xk1), (xk2 - xk1) * (yk2 - yk1), torch.zeros_like(x1))
    union = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - inter
    return inter, union
def giou_loss(b1, b2, eps=1e-7):
    inter, union = inter_union(b1, b2); iou = inter / (union + eps)
    x1, y1, x2, y2 = b1.unbind(-1); x1g, y1g, x2g, y2g = b2.unbind(-1)
    area_c = (torch.maximum(x2, x2g) - torch.minimum(x1, x1g)) * (torch.maximum(y2, y2g) - torch.minimum(y1, y1g))
    return 1 - (iou - (area_c - union) / (area_c + eps))
def diou_loss(b1, b2, eps=1e-7):
    inter, union = inter_union(b1, b2); iou = inter / (union + eps)
    x1, y1, x2, y2 = b1.unbind(-1); x1g, y1g, x2g, y2g = b2.unbind(-1)
    diag = (torch.maximum(x2, x2g) - torch.minimum(x1, x1g)) ** 2 + (torch.maximum(y2, y2g) - torch.minimum(y1, y1g)) ** 2 + eps
    cd = ((x2 + x1) / 2 - (x1g + x2g) / 2) ** 2 + ((y2 + y1) / 2 - (y1g + y2g) / 2) ** 2
    return 1 - iou + cd / diag, iou
def ciou_loss(b1, b2, eps=1e-7):
    dl, iou = diou_loss(b1, b2)
    x1, y1, x2, y2 = b1.unbind(-1); x1g, y1g, x2g, y2g = b2.unbind(-1)
    v = (4 / math.pi ** 2) * (torch.atan((x2g - x1g) / (y2g - y1g)) - torch.atan((x2 - x1) / (y2 - y1))) ** 2
    return dl + v / (1 - iou + v + eps) * v

def bilinear_cuda(x, py, px):
    """x (1, C, H, W); py, px (P,) -> (C, P): the bilinear sample of torchvision's roi_align kernel (points
    within one pixel outside the map are clamped to the border, farther ones read 0)"""
    _, C, H, W = x.shape
    inside = ((py >= -1) & (py <= H) & (px >= -1) & (px <= W)).float()
    py = torch.clamp(py, min=0); px = torch.clamp(px, min=0)
    yl = torch.floor(py); xl = torch.floor(px)
    yl = torch.where(yl >= H - 1, torch.full_like(yl, H - 1), yl); xl = torch.where(xl >= W - 1, torch.full_like(xl, W - 1), xl)
    py = torch.where(py > H - 1, torch.full_like(py, H - 1), py); px = torch.where(px > W - 1, torch.full_like(px, W - 1), px)
    yh = torch.clamp(yl + 1, max=H - 1); xh = torch.clamp(xl + 1, max=W - 1)
    ly = py - yl; lx = px - xl; hy = 1 - ly; hx = 1 - lx
    ah = torch.arange(H, dtype=x.dtype); aw = torch.arange(W, dtype=x.dtype)
    def oh(v, a): return (v[:, None] == a[None, :]).to(x.dtype)                 # (P, n)
    img = x[0]
    def samp(oy, ox): return torch.einsum('chw,ph,pw->cp', img, oy, ox)
    v = hy * hx * samp(oh(yl, ah), oh(xl, aw)) + hy * lx * samp(oh(yl, ah), oh(xh, aw)) + ly * hx * samp(oh(yh, ah), oh(xl, aw)) + ly * lx * samp(oh(yh, ah), oh(xh, aw))
    return v * inside[None, :]
def roi_align(x, rois, k, scale, sr, aligned):
    """torchvision.ops.roi_align for a (1, C, H, W) map and R rois (batch index 0), sr x sr samples per bin"""
    off = 0.5 if aligned else 0.0; outs = []
    for r in range(rois.shape[0]):
        x1 = rois[r, 1] * scale - off; y1 = rois[r, 2] * scale - off; x2 = rois[r, 3] * scale - off; y2 = rois[r, 4] * scale - off
        rw = x2 - x1; rh = y2 - y1
        if not aligned: rw = torch.clamp(rw, min=1.0); rh = torch.clamp(rh, min=1.0)
        bw = rw / k; bh = rh / k
        py = torch.stack([y1 + i * bh + (a + 0.5) * bh / sr for i in range(k) for j in range(k) for a in range(sr) for b in range(sr)])
        px = torch.stack([x1 + j * bw + (b + 0.5) * bw / sr for i in range(k) for j in range(k) for a in range(sr) for b in range(sr)])
        v = bilinear_cuda(x, py, px).reshape(-1, k, k, sr * sr).mean(-1)
        outs.append(v)
    return torch.stack(outs)

def add_ops(case, rcase, R):
    torch.manual_seed(3)
    ar8 = torch.arange(8).float()
    boxes = torch.tensor([[2., 3., 12., 14.], [3., 4., 13., 15.], [20., 20., 30., 32.], [1., 1., 6., 6.], [22., 18., 31., 30.], [40., 40., 50., 52.], [41., 42., 49., 50.], [0., 30., 8., 40.]])
    boxes2 = boxes + torch.tensor([1., -1., 2., 1.])
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.85, 0.5, 0.95, 0.4])
    # ---- box utilities
    case('box_area', lambda s, x: O.box_area(x), boxes)
    case('box_convert', lambda s, x: O.box_convert(x, 'xyxy', 'cxcywh'), boxes)
    case('box_iou', lambda s, x: O.box_iou(x, s.y), boxes, y=boxes2)
    case('generalized_box_iou', lambda s, x: O.generalized_box_iou(x, s.y), boxes, y=boxes2)
    case('distance_box_iou', lambda s, x: O.distance_box_iou(x, s.y), boxes, y=boxes2)
    case('complete_box_iou', lambda s, x: O.complete_box_iou(x, s.y), boxes, y=boxes2)
    case('clip_boxes_to_image', lambda s, x: O.clip_boxes_to_image(x, (32, 48)), boxes)
    rcase('remove_small_boxes', lambda s, x: ((x[:, 2] - x[:, 0] >= 8) & (x[:, 3] - x[:, 1] >= 8)).float(),
          lambda s, x: (lambda k: (torch.arange(8)[:, None] == k[None, :]).any(1).float())(O.remove_small_boxes(x, 8)), boxes)
    ys, xs = torch.meshgrid(torch.arange(16).float(), torch.arange(16).float(), indexing='ij')
    masks = torch.zeros(3, 16, 16); masks[0, 2:9, 3:12] = 1; masks[1, 10:15, 0:5] = 1; masks[2, 5:6, 7:8] = 1
    rcase('masks_to_boxes', lambda s, x: masks_to_boxes(x, s.ys, s.xs), lambda s, x: O.masks_to_boxes(x.bool()), masks, ys=ys, xs=xs)
    rcase('nms', lambda s, x: nms_mask(x, s.sc, 0.5, s.ar), lambda s, x: (lambda k: (torch.arange(8)[:, None] == k[None, :]).any(1).float())(O.nms(x, s.sc, 0.5)), boxes, sc=scores, ar=ar8)
    idxs = torch.tensor([0, 0, 1, 0, 1, 2, 2, 1])
    rcase('batched_nms', lambda s, x: nms_mask(x + s.off[:, None], s.sc, 0.5, s.ar), lambda s, x: (lambda k: (torch.arange(8)[:, None] == k[None, :]).any(1).float())(O.batched_nms(x, s.sc, s.idx, 0.5)),
          boxes, sc=scores, ar=ar8, off=idxs.float() * 100.0, idx=idxs)
    # ---- losses (sum-reduced scalars as 1-element tensors)
    case('sigmoid_focal_loss', lambda s, x: O.sigmoid_focal_loss(x, s.t, reduction='sum').reshape(1), R(4, 8), t=(torch.rand(4, 8) > 0.5).float())
    # (the IoU losses' _loss_inter_union assigns through a boolean mask, which fails in torch-mlir's
    # lowering: the same formulas with torch.where)
    rcase('generalized_box_iou_loss', lambda s, x: giou_loss(x, s.y).sum().reshape(1), lambda s, x: O.generalized_box_iou_loss(x, s.y, reduction='sum').reshape(1), boxes, y=boxes2)
    rcase('distance_box_iou_loss', lambda s, x: diou_loss(x, s.y)[0].sum().reshape(1), lambda s, x: O.distance_box_iou_loss(x, s.y, reduction='sum').reshape(1), boxes, y=boxes2)
    rcase('complete_box_iou_loss', lambda s, x: ciou_loss(x, s.y).sum().reshape(1), lambda s, x: O.complete_box_iou_loss(x, s.y, reduction='sum').reshape(1), boxes, y=boxes2)
    # ---- RoI operators on a 1 x C x 16 x 16 feature map, 3 constant RoIs (batch index 0), output 2 x 2
    feat = R(1, 4, 16, 16); rois = torch.tensor([[0., 2., 3., 12., 14.], [0., 5., 1., 15., 9.], [0., 0., 0., 15., 15.]])
    rcase('roi_align', lambda s, x: roi_align(x, s.rois, 2, 0.5, 2, False), lambda s, x: O.roi_align(x, s.rois, 2, 0.5, 2, False), feat, rois=rois)
    rcase('roi_align_aligned', lambda s, x: roi_align(x, s.rois, 2, 1.0, 2, True), lambda s, x: O.roi_align(x, s.rois, 2, 1.0, 2, True), feat, rois=rois)
    rcase('roi_pool', lambda s, x: torch.amax(x[0][None, :, None, None] * s.M[:, None] - 1e6 * (1 - s.M[:, None]), dim=(-2, -1)), lambda s, x: O.roi_pool(x, s.rois, 2, 1.0), feat, rois=rois, M=roi_pool_masks(rois, 16, 16, 2, 1.0))
    feat16 = R(1, 16, 16, 16)   # C * k * k = 4 * 2 * 2
    rois_s = torch.tensor([[0., 2., 3., 9., 10.], [0., 5., 1., 12., 8.], [0., 0., 0., 9., 9.]])   # ps_roi_pool_ref reads a 6-cell window per bin: rois up to 10 px for k = 2
    rcase('ps_roi_align', lambda s, x: ps_roi_align(x, s.rois, 2, 1.0, 2), lambda s, x: O.ps_roi_align(x, s.rois, 2, 1.0, 2), feat16, rois=rois)
    rcase('ps_roi_pool', lambda s, x: dcn_ops.ps_roi_pool_ref(x, s.rois, 2, 1.0), lambda s, x: O.ps_roi_pool(x, s.rois, 2, 1.0), feat16, rois=rois_s)
    ra = O.RoIAlign(2, 0.5, 2, aligned=False); rp = O.RoIPool(2, 1.0); psa = O.PSRoIAlign(2, 1.0, 2); psp = O.PSRoIPool(2, 1.0)
    rcase('roi_align_module', lambda s, x: roi_align(x, s.rois, 2, 0.5, 2, False), lambda s, x: ra(x, s.rois), feat, rois=rois)
    rcase('roi_pool_module', lambda s, x: torch.amax(x[0][None, :, None, None] * s.M[:, None] - 1e6 * (1 - s.M[:, None]), dim=(-2, -1)), lambda s, x: rp(x, s.rois), feat, rois=rois, M=roi_pool_masks(rois, 16, 16, 2, 1.0))
    rcase('ps_roi_align_module', lambda s, x: ps_roi_align(x, s.rois, 2, 1.0, 2), lambda s, x: psa(x, s.rois), feat16, rois=rois)
    rcase('ps_roi_pool_module', lambda s, x: dcn_ops.ps_roi_pool_ref(x, s.rois, 2, 1.0), lambda s, x: psp(x, s.rois), feat16, rois=rois_s)
    msra = O.MultiScaleRoIAlign(['0'], 2, 2)   # one level, the scale inferred from the 64x64 image (16 / 64 = 1/4)
    rcase('multi_scale_roi_align', lambda s, x: roi_align(x, s.rois, 2, 0.25, 2, False), lambda s, x: msra({'0': x}, [s.rois[:, 1:]], [(64, 64)]), feat, rois=rois)
    # ---- deformable convolution (DCN v1 / v2 with a modulation mask) and the module
    w = R(6, 4, 3, 3) * 0.2; b = R(6) * 0.1; off = R(1, 18, 16, 16) * 0.5; mk = torch.sigmoid(R(1, 9, 16, 16))
    rcase('deform_conv2d', lambda s, x: dcn_ops.deform_conv2d_ref(x, s.off, s.w, s.b, padding=1), lambda s, x: O.deform_conv2d(x, s.off, s.w, s.b, padding=1), feat, off=off, w=w, b=b)
    rcase('deform_conv2d_mask', lambda s, x: dcn_ops.deform_conv2d_ref(x, s.off, s.w, s.b, padding=1, mask=s.mk), lambda s, x: O.deform_conv2d(x, s.off, s.w, s.b, padding=1, mask=s.mk), feat, off=off, w=w, b=b, mk=mk)
    dcm = O.DeformConv2d(4, 6, 3, padding=1)
    rcase('deform_conv2d_module', lambda s, x: dcn_ops.deform_conv2d_ref(x, s.off, s.w, s.b, padding=1), lambda s, x: dcm(x, s.off), feat, off=off, w=dcm.weight.detach().clone(), b=dcm.bias.detach().clone())
    # ---- layers (eval mode; the stochastic ones are the identity in eval)
    def mod(n, m, x, out=None):
        case(n, (lambda m, out: lambda s, x: out(m(x)) if out else m(x))(m.eval(), out), x)
    mod('conv2d_norm_activation', O.Conv2dNormActivation(4, 8, 3), feat)
    mod('conv3d_norm_activation', O.Conv3dNormActivation(2, 4, 3), R(1, 2, 6, 6, 6))
    fbn = O.FrozenBatchNorm2d(4); fbn.weight.copy_(torch.rand(4) + 0.5); fbn.bias.copy_(R(4)); fbn.running_mean.copy_(R(4)); fbn.running_var.copy_(torch.rand(4) + 0.5)
    mod('frozen_batch_norm2d', fbn, feat)
    mod('mlp', O.MLP(8, [16, 4]), R(2, 8))
    mod('permute', O.Permute([0, 2, 3, 1]), feat)
    mod('squeeze_excitation', O.SqueezeExcitation(4, 2), feat)
    mod('drop_block2d', O.DropBlock2d(0.3, 3), feat); mod('drop_block3d', O.DropBlock3d(0.3, 3), R(1, 2, 6, 6, 6))
    mod('stochastic_depth', O.StochasticDepth(0.3, 'row'), feat)
    case('drop_block2d_fn', lambda s, x: O.drop_block2d(x, 0.3, 3, training=False), feat)
    case('drop_block3d_fn', lambda s, x: O.drop_block3d(x, 0.3, 3, training=False), R(1, 2, 6, 6, 6))
    case('stochastic_depth_fn', lambda s, x: O.stochastic_depth(x, 0.3, 'row', training=False), feat)
    fpn = O.FeaturePyramidNetwork([4, 4], 4).eval()
    case('feature_pyramid_network', lambda s, x: (lambda o: torch.cat([o['0'].flatten(), o['1'].flatten()]))(fpn({'0': x, '1': NF.avg_pool2d(x, 2)})), feat)
