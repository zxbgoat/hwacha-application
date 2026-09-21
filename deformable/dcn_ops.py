"""Deformable ConvNets operators written with plain tensor ops (gather + bilinear + matmul), so that
torch-mlir can lower them; numerically checked against torchvision.ops.deform_conv2d / ps_roi_pool.
Shapes are static and the sampling grid is built from the offsets, as in the paper (DCN v1: offsets,
DCN v2: offsets + modulation mask)."""
import torch, torch.nn as nn, torch.nn.functional as F

def bilinear_gather(x, py, px):
    """x: (N,C,H,W); py, px: (N,P) sample positions (float, in pixels) -> (N,C,P) bilinear samples with
    zero padding outside the image (same convention as deform_conv2d)."""
    N, C, H, W = x.shape
    y0 = torch.floor(py); x0 = torch.floor(px); y1 = y0 + 1; x1 = x0 + 1
    wy1 = py - y0; wx1 = px - x0; wy0 = 1 - wy1; wx0 = 1 - wx1
    flat = x.reshape(N, C, H * W)
    def tap(yy, xx, w):
        valid = ((yy >= 0) & (yy <= H - 1) & (xx >= 0) & (xx <= W - 1)).to(x.dtype)
        idx = (yy.clamp(0, H - 1) * W + xx.clamp(0, W - 1)).long()            # (N,P)
        v = torch.gather(flat, 2, idx.unsqueeze(1).expand(N, C, idx.shape[1]))  # (N,C,P)
        return v * (w * valid).unsqueeze(1)
    return tap(y0, x0, wy0 * wx0) + tap(y0, x1, wy0 * wx1) + tap(y1, x0, wy1 * wx0) + tap(y1, x1, wy1 * wx1)

def deform_conv2d_ref(x, offset, weight, bias=None, stride=1, padding=0, dilation=1, mask=None):
    """Same semantics as torchvision.ops.deform_conv2d (groups=1, offset_groups=1).
    offset: (N, 2*kh*kw, Ho, Wo) laid out as [dy0, dx0, dy1, dx1, ...]; mask: (N, kh*kw, Ho, Wo)."""
    N, C, H, W = x.shape; O, _, kh, kw = weight.shape
    Ho = (H + 2 * padding - dilation * (kh - 1) - 1) // stride + 1
    Wo = (W + 2 * padding - dilation * (kw - 1) - 1) // stride + 1
    K = kh * kw
    oy = torch.arange(Ho, dtype=x.dtype).view(Ho, 1) * stride - padding
    ox = torch.arange(Wo, dtype=x.dtype).view(1, Wo) * stride - padding
    ky = torch.arange(kh, dtype=x.dtype).view(kh, 1).expand(kh, kw).reshape(K) * dilation
    kx = torch.arange(kw, dtype=x.dtype).view(1, kw).expand(kh, kw).reshape(K) * dilation
    off = offset.reshape(N, K, 2, Ho, Wo)
    py = (oy.view(1, 1, Ho, 1) + ky.view(1, K, 1, 1) + off[:, :, 0]).reshape(N, K * Ho * Wo)
    px = (ox.view(1, 1, 1, Wo) + kx.view(1, K, 1, 1) + off[:, :, 1]).reshape(N, K * Ho * Wo)
    cols = bilinear_gather(x, py, px).reshape(N, C, K, Ho * Wo)             # im2col with deformed taps
    if mask is not None: cols = cols * mask.reshape(N, 1, K, Ho * Wo)
    y = torch.matmul(weight.reshape(O, C * K), cols.reshape(N, C * K, Ho * Wo))   # (N,O,Ho*Wo)
    if bias is not None: y = y + bias.view(1, O, 1)
    return y.reshape(N, O, Ho, Wo)

def ps_roi_pool_ref(x, rois, output_size, spatial_scale=1.0):
    """Position-sensitive RoI pooling (R-FCN), exactly torchvision.ops.ps_roi_pool: x (1, C*k*k, H, W),
    rois (R, 5) [batch, x1, y1, x2, y2] -> (R, C, k, k). Bin (i, j) of roi r averages the integer cells
    floor(y1 + i*bh) .. ceil(y1 + (i+1)*bh) - 1 (same for x) of channel group c*k*k + i*k + j, where
    bh = (y2 - y1 + 1) / k. The bin bounds are data-dependent; for static shapes every bin reads a fixed
    window of MAXB x MAXB candidate cells starting at its floor and masks the cells past its end, so the
    result is exact as long as no bin spans more than MAXB cells (rois up to (MAXB-1)*k pixels)."""
    k = output_size; R = rois.shape[0]; _, CK, H, W = x.shape; C = CK // (k * k)
    MAXB = 6
    rnd = lambda t: torch.floor(t + 0.5)   # torch.round lowers to llvm.rint, which hwacha-cc has no kernel for
    x1 = rnd(rois[:, 1] * spatial_scale); y1 = rnd(rois[:, 2] * spatial_scale)
    x2 = rnd(rois[:, 3] * spatial_scale); y2 = rnd(rois[:, 4] * spatial_scale)
    bw = (x2 - x1).clamp(min=1) / k; bh = (y2 - y1).clamp(min=1) / k                           # (R,) torchvision: roi_w = max(x2 - x1, 1)
    ib = torch.arange(k, dtype=x.dtype)
    ceil = lambda t: -torch.floor(-t)   # hwacha-cc lowers floorf, not ceilf
    ys = torch.floor(y1.view(R, 1) + ib.view(1, k) * bh.view(R, 1)); ye = ceil(y1.view(R, 1) + (ib.view(1, k) + 1) * bh.view(R, 1))   # (R,k)
    xs = torch.floor(x1.view(R, 1) + ib.view(1, k) * bw.view(R, 1)); xe = ceil(x1.view(R, 1) + (ib.view(1, k) + 1) * bw.view(R, 1))
    ys = ys.clamp(0, H); ye = ye.clamp(0, H); xs = xs.clamp(0, W); xe = xe.clamp(0, W)
    d = torch.arange(MAXB, dtype=x.dtype)
    cy = ys.view(R, k, 1) + d.view(1, 1, MAXB); cx = xs.view(R, k, 1) + d.view(1, 1, MAXB)          # candidate cells (R,k,MAXB)
    my = (cy < ye.view(R, k, 1)).to(x.dtype); mx = (cx < xe.view(R, k, 1)).to(x.dtype)
    cy = cy.clamp(0, H - 1); cx = cx.clamp(0, W - 1)
    # gather the window of every (roi, bin i, bin j): (R, k, k, MAXB, MAXB)
    idx = (cy.view(R, k, 1, MAXB, 1) * W + cx.view(R, 1, k, 1, MAXB)).expand(R, k, k, MAXB, MAXB).reshape(R, -1).long()
    m = (my.view(R, k, 1, MAXB, 1) * mx.view(R, 1, k, 1, MAXB)).expand(R, k, k, MAXB, MAXB)     # cell inside the bin
    flat = x.reshape(CK, H * W)
    v = flat.index_select(1, idx.reshape(-1)).reshape(CK, R, -1).transpose(0, 1)                   # (r, CK, P): CK = (c, ci, cj), P = (bi, bj, dy, dx)
    v = v.reshape(R, C, k, k, k, k, MAXB, MAXB)                                                    # (r, c, ci, cj, bi, bj, dy, dx)
    # keep channel group (i,j) for bin (i,j): the (ci, bi) and (cj, bj) diagonals (an einsum with two
    # identity matrices lowers to batch_matmuls that hwacha-mlir unrolls into millions of instructions)
    v = torch.diagonal(v, dim1=2, dim2=4)                                                          # (r, c, cj, bj, dy, dx, i)
    v = torch.diagonal(v, dim1=2, dim2=3)                                                          # (r, c, dy, dx, i, j)
    v = v.permute(0, 1, 4, 5, 2, 3)                                                                # (r, c, i, j, dy, dx)
    cnt = m.sum(dim=(3, 4)).clamp(min=1)                                                           # (R,k,k)
    return (v * m.view(R, 1, k, k, MAXB, MAXB)).sum(dim=(4, 5)) / cnt.view(R, 1, k, k)


def deform_ps_roi_pool_ref(x, rois, output_size, offsets, spatial_scale=1.0, trans_std=0.1):
    """Deformable PS-RoI pooling (Deformable ConvNets, sec. 2.3): bin (i,j) of roi r is shifted by
    gamma * (roi_w, roi_h) * offsets[r, :, i, j] and its value is the bilinear-sampled average of the
    shifted bin, on the position-sensitive channel group c*k*k + i*k + j. offsets: (R, 2, k, k) as
    (dy, dx) in normalized units. Each bin is sampled on a fixed S x S grid (S=2, as in the reference
    implementation's sample_per_part) with bilinear interpolation and zero padding, like deform_conv."""
    k = output_size; R = rois.shape[0]; _, CK, H, W = x.shape; C = CK // (k * k); S = 2
    rnd = lambda t: torch.floor(t + 0.5)
    x1 = rnd(rois[:, 1] * spatial_scale) - 0.5; y1 = rnd(rois[:, 2] * spatial_scale) - 0.5
    x2 = rnd(rois[:, 3] * spatial_scale) + 0.5; y2 = rnd(rois[:, 4] * spatial_scale) + 0.5
    rw = (x2 - x1).clamp(min=0.1); rh = (y2 - y1).clamp(min=0.1)                         # (R,)
    bw = rw / k; bh = rh / k
    ib = torch.arange(k, dtype=x.dtype); t = (torch.arange(S, dtype=x.dtype) + 0.5) / S
    # sample positions of bin (i,j): (R, k, k, S, S)
    py = y1.view(R, 1, 1, 1, 1) + (ib.view(1, k, 1, 1, 1) + t.view(1, 1, 1, S, 1)) * bh.view(R, 1, 1, 1, 1) + offsets[:, 0].view(R, k, k, 1, 1) * trans_std * rh.view(R, 1, 1, 1, 1)
    px = x1.view(R, 1, 1, 1, 1) + (ib.view(1, 1, k, 1, 1) + t.view(1, 1, 1, 1, S)) * bw.view(R, 1, 1, 1, 1) + offsets[:, 1].view(R, k, k, 1, 1) * trans_std * rw.view(R, 1, 1, 1, 1)
    py = py.expand(R, k, k, S, S).reshape(R, -1); px = px.expand(R, k, k, S, S).reshape(R, -1)
    xs = x.expand(R, CK, H, W)
    v = bilinear_gather(xs, py, px).reshape(R, C, k, k, k, k, S, S)                     # (r, c, ci, cj, bi, bj, sy, sx)
    v = torch.diagonal(v, dim1=2, dim2=4); v = torch.diagonal(v, dim1=2, dim2=3)         # (r, c, sy, sx, i, j)
    return v.mean(dim=(2, 3))                                                            # (R, C, k, k)
