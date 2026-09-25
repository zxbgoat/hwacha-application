"""torchvision.io cases (docs.pytorch.org/vision/stable/io.html) for export_tvi.py. The page is codecs
and file I/O; the cases cover the parts that are computation: the baseline JPEG encode / decode
(encode_jpeg + decode_jpeg: RGB -> YCbCr, 4:2:0 chroma subsampling with libjpeg's rounding, 8x8 DCT by
matmul, quality-75 quantisation, dequantisation, IDCT, libjpeg's "fancy" triangle chroma upsampling,
YCbCr -> RGB), the decoders' ImageReadMode conversions (GRAY = libjpeg's / libpng's luma weights,
RGB_ALPHA = an opaque alpha plane) and the lossless PNG round trip (the identity). The reference is
the genuine torchvision codec call on the bytes; the JPEG re-implementation uses float DCTs where
libjpeg uses its integer "islow" ones, so it agrees to within a couple of levels of 255 (the host
tolerance), not bit-exactly. Frames go in and out as float tensors of 0..255 values."""
import math, torch, torchvision.io as IO
from torchvision.io import ImageReadMode as M

def dct_matrix():
    D = torch.zeros(8, 8)
    for u in range(8):
        for x in range(8): D[u, x] = math.sqrt(0.25 if u else 0.125) * math.cos((2 * x + 1) * u * math.pi / 16)
    return D
LUMA = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]
CHROMA = [17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99] + [99] * 32
def qtable(base, quality):
    s = 5000 // quality if quality < 50 else 200 - 2 * quality
    return torch.tensor([min(max((b * s + 50) // 100, 1), 255) for b in base], dtype=torch.float32).reshape(8, 8)
def rnd(x): return torch.floor(x + 0.5)
def rnd_away(x): return torch.sign(x) * torch.floor(x.abs() + 0.5)
def blocks(p):   # (H, W) -> (nb, 8, 8)
    H, W = p.shape; return p.reshape(H // 8, 8, W // 8, 8).permute(0, 2, 1, 3).reshape(-1, 8, 8)
def unblocks(b, H, W): return b.reshape(H // 8, W // 8, 8, 8).permute(0, 2, 1, 3).reshape(H, W)
def rgb_to_ycc(x):
    """libjpeg's jccolor: 16-bit fixed-point weights (19595 + 38470 + 7471 = 65536), + ONE_HALF, >> 16"""
    r, g, b = x[0], x[1], x[2]
    y = torch.floor((19595 * r + 38470 * g + 7471 * b + 32768) / 65536)
    cb = torch.floor((-11059 * r - 21709 * g + 32768 * b + 128 * 65536 + 32767) / 65536)   # CBCR_OFFSET + ONE_HALF - 1
    cr = torch.floor((32768 * r - 27439 * g - 5329 * b + 128 * 65536 + 32767) / 65536)
    return y, cb, cr
def png_gray(x):
    """libpng's rgb_to_gray as torchvision configures it: 15-bit fixed-point weights 9794 / 19234 / 3740 (0.2989,
    0.5870, the rest), truncating (fitted exactly against decode_png(mode=GRAY) on random images)"""
    return torch.floor((9794 * x[:1] + 19234 * x[1:2] + 3740 * x[2:3]) / 32768)
def ycc_to_rgb(y, cb, cr):
    cb = cb - 128; cr = cr - 128
    return torch.stack([rnd(y + 1.402 * cr), rnd(y - 0.344136 * cb - 0.714136 * cr), rnd(y + 1.772 * cb)]).clamp(0, 255)
def downsample(c, bias):
    """libjpeg's h2v2_downsample: (sum of the 2x2 + bias) >> 2, the bias alternating 1, 2 along the row"""
    s = c[0::2, 0::2] + c[0::2, 1::2] + c[1::2, 0::2] + c[1::2, 1::2]
    return torch.floor((s + bias) / 4)
def upsample(c, colbias):
    """libjpeg's h2v2_fancy_upsample (triangle filter): vertically 3 * nearer + further, then horizontally
    (3 * this + neighbour + 8 or 7) >> 4; edges replicate"""
    h, w = c.shape
    up = torch.cat([c[:1], c], 0); dn = torch.cat([c, c[-1:]], 0)
    rows = torch.stack([3 * c + up[:-1], 3 * c + dn[1:]], 1).reshape(2 * h, w)        # (2h, w) column sums
    lf = torch.cat([rows[:, :1], rows[:, :-1]], 1); rt = torch.cat([rows[:, 1:], rows[:, -1:]], 1)
    out = torch.stack([3 * rows + lf + 8, 3 * rows + rt + 7], 2).reshape(2 * h, 2 * w)
    out = out + colbias
    return torch.floor(out / 16)
def jpeg_roundtrip(x, D, ql, qc, bias, colbias, gray=False):
    """x (3, H, W) float 0..255 (H, W multiples of 16) -> the decoded (3, H, W) of libjpeg's baseline 4:2:0
    (gray: the decoded Y plane, what decode_jpeg(mode=GRAY) returns)"""
    H, W = x.shape[1:]
    y, cb, cr = rgb_to_ycc(x)
    planes = []
    for p, q, sub in ((y, ql, False), (cb, qc, True), (cr, qc, True)):
        if sub: p = downsample(p, bias)
        b = blocks(p - 128)
        coef = rnd_away((D @ b @ D.t()) / q)                   # quantised coefficients (the "file")
        rec = D.t() @ (coef * q) @ D + 128
        rec = rnd(rec).clamp(0, 255)
        rec = unblocks(rec, *p.shape)
        if sub: rec = upsample(rec, colbias).clamp(0, 255)
        planes.append(rec)
    return planes[0][None] if gray else ycc_to_rgb(*planes)

def add_io(case, rcase, R):
    torch.manual_seed(5)
    H = W = 16
    img8 = (torch.rand(3, H, W) * 255).to(torch.uint8); imgf = img8.float()
    # smooth content too (a gradient plus texture): JPEG's typical input
    yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing='ij')
    smooth = torch.stack([8 * xx + 60, 6 * yy + 40, 120 + 40 * torch.sin(xx / 3) + 20 * torch.cos(yy / 2)]).clamp(0, 255)
    smooth8 = torch.floor(smooth).to(torch.uint8); smoothf = smooth8.float()
    D = dct_matrix(); ql = qtable(LUMA, 75); qc = qtable(CHROMA, 75)
    bias = torch.tensor([1., 2.] * (W // 4)).repeat(H // 2, 1); colbias = torch.zeros(H, W)
    def jref(x, mode=M.UNCHANGED): return IO.decode_jpeg(IO.encode_jpeg(x.to(torch.uint8), quality=75), mode=mode).float()
    J = dict(D=D, ql=ql, qc=qc, bias=bias, colbias=colbias, _atol=2.5)   # float vs libjpeg's integer DCTs: within 2 levels
    for nm, im in (('encode_decode_jpeg', smoothf), ('encode_decode_jpeg_noise', imgf)):
        rcase(nm, lambda s, x: jpeg_roundtrip(x, s.D, s.ql, s.qc, s.bias, s.colbias), lambda s, x: jref(x), im, **J)
    rcase('decode_jpeg_gray', lambda s, x: jpeg_roundtrip(x, s.D, s.ql, s.qc, s.bias, s.colbias, gray=True), lambda s, x: jref(x, M.GRAY), smoothf, **J)
    def pref(x, mode=M.UNCHANGED): return IO.decode_png(IO.encode_png(x.to(torch.uint8)), mode=mode).float()
    rcase('encode_decode_png', lambda s, x: x, lambda s, x: pref(x), imgf)
    rcase('decode_png_gray', lambda s, x: png_gray(x), lambda s, x: pref(x, M.GRAY), imgf)
    rcase('decode_png_rgb_alpha', lambda s, x: torch.cat([x, torch.ones_like(x[:1]) * 255]), lambda s, x: pref(x, M.RGB_ALPHA), imgf)
    rcase('decode_image_rgb', lambda s, x: x.expand(3, H, W).clone(), lambda s, x: IO.decode_image(IO.encode_png(x.to(torch.uint8)), mode=M.RGB).float(), imgf[:1].clone())
    def rimg(x):
        import tempfile, os
        p = os.path.join(tempfile.gettempdir(), 'tvintf_read_image.png'); IO.write_png(x.to(torch.uint8), p); return IO.read_image(p, mode=M.GRAY).float()
    rcase('write_png_read_image', lambda s, x: png_gray(x), lambda s, x: rimg(x), imgf)
