#!/usr/bin/env python3
"""Export one top-level torch.* tensor function (docs.pytorch.org/docs/2.14/torch.html) through torch-mlir
to linalg-on-tensors, plus a PyTorch reference for one call. Every case is a single-input module
net(x) -> one float tensor (second operands are constant buffers, integer results are cast to float),
so the generic host mod_main.c drives them all.       usage: export_intf.py <fn> <mlir_out> <check_out>"""
import sys, struct, numpy as np, torch
from torch_mlir import fx

class Case(torch.nn.Module):
    """net(x) = body(self, x); every tensor in `bufs` becomes a constant buffer of the module"""
    def __init__(s, body, **bufs):
        super().__init__(); s.body = body
        for k, v in bufs.items(): s.register_buffer(k, v)
    def forward(s, x): return s.body(s, x)

class Ref(Case):
    """exported body differs from the reference: the reference (what the check.bin holds) is computed
    with the genuine torch call, the exported graph with an equivalent op composition that torch-mlir
    lowers to linalg (the original op lowers to a tm_tensor op hwacha-mlir does not take)."""
    def __init__(s, body, ref, **bufs): super().__init__(body, **bufs); s.ref = ref
    def reference(s, x): return s.ref(s, x)

def R(*shape): return torch.randn(*shape)

def build(fn):
    torch.manual_seed(0)
    C = {}
    def case(name, body, x, **bufs): C[name] = lambda: (Case(body, **bufs), x)
    def rcase(name, body, ref, x, **bufs): C[name] = lambda: (Ref(body, ref, **bufs), x)
    # ---- torch.topk(x, k): the k largest of every row, descending, with their indices; the result is
    # [values | indices] (B, 2k). aten.topk lowers to tm_tensor.sort, so the exported graph ranks every
    # element by the number of strictly greater elements in its row (random data: no ties) and picks
    # the element of rank i by a one-hot sum: values_i = sum_j x_j [rank_j == i], indices_i = sum_j j [..].
    N, K = 16, 4
    def topk_body(s, x):
        gt = (x[:, :, None] < x[:, None, :]).to(x.dtype)                   # gt[b, j, m] = x_m > x_j
        rank = gt.sum(-1)                                                   # (B, N): 0 = row maximum
        sel = (rank[:, None, :] == s.r[None, :, None]).to(x.dtype)         # (B, 2K, N): rank == i mod K
        w = x[:, None, :] * s.isval[None, :, None] + s.idx[None, None, :] * (1 - s.isval)[None, :, None]
        return (sel * w).sum(-1)                                            # (B, 2K)
    def topk_ref(s, x):
        v, i = torch.topk(x, K)
        return torch.cat([v, i.to(x.dtype)], -1)
    rcase('topk', topk_body, topk_ref, R(4, N), r=(torch.arange(2 * K) % K).float(), isval=(torch.arange(2 * K) < K).float(),
          idx=torch.arange(N).float())
    # ---- torch.fft (docs.pytorch.org/docs/2.14/fft.html). torch-mlir has no complex tensors, so a
    # complex tensor is a real one with [real | imag] concatenated along the last dim (a real input /
    # output keeps its shape), and a DFT along a dim is a matmul with a constant block matrix:
    #   along the last dim:   [xr | xi] @ [[Fr, Fi], [-Fi, Fr]]      (real input: x @ [Fr | Fi])
    #   along another dim:    Fr @ x + Fi @ (x @ P),  P = [[0, I], [-I, 0]] turns [xr | xi] into [-xi | xr]
    # with F = C - iS (forward) or (C + iS) / N (inverse), C, S = cos / sin(2 pi n k / N). rfft keeps the
    # first N/2+1 columns; irfft is the C2R sum with the Hermitian weights (1, 2, ..., 2, 1) / N; hfft(x)
    # = N * irfft(conj x), ihfft(x) = conj(rfft x) / N (the n-D ones likewise over all dims). The
    # reference is the genuine torch.fft call (complex parts split / joined the same way).
    import math
    def dft(N, inverse):
        n = torch.arange(N, dtype=torch.float64); ang = 2 * math.pi * torch.outer(n, n) / N
        Cm, Sm = torch.cos(ang), torch.sin(ang)
        return ((Cm / N).float(), (Sm / N).float()) if inverse else (Cm.float(), (-Sm).float())
    def bcomplex(Fr, Fi): return torch.cat([torch.cat([Fr, Fi], 1), torch.cat([-Fi, Fr], 1)], 0)   # right factor, complex input
    def breal(Fr, Fi): return torch.cat([Fr, Fi], 1)                                                  # right factor, real input
    def swapP(N): I = torch.eye(N); Z = torch.zeros(N, N); return torch.cat([torch.cat([Z, I], 1), torch.cat([-I, Z], 1)], 0)
    def rfft_mat(N):
        Fr, Fi = dft(N, False); H = N // 2 + 1; return torch.cat([Fr[:, :H], Fi[:, :H]], 1)          # (N, 2H)
    def irfft_mat(N):
        H = N // 2 + 1; k = torch.arange(H, dtype=torch.float64); n = torch.arange(N, dtype=torch.float64)
        ang = 2 * math.pi * torch.outer(k, n) / N; w = torch.full((H, 1), 2.0, dtype=torch.float64); w[0] = 1; w[H - 1] = 1
        return torch.cat([w * torch.cos(ang) / N, -w * torch.sin(ang) / N], 0).float()             # (2H, N)
    def conjmask(H): return torch.cat([torch.ones(H), -torch.ones(H)])
    def along(x, dim, Fr, Fi, P):
        """complex DFT along `dim` (not the last) of x in [xr | xi] layout: Fr @ x + Fi @ (x @ P)"""
        Q = x @ P
        if dim == -2: return Fr @ x + Fi @ Q
        shp = x.shape; x2 = x.reshape(*shp[:dim + 1], -1); Q2 = Q.reshape(*shp[:dim + 1], -1)
        return (Fr @ x2 + Fi @ Q2).reshape(shp)
    def cplx(x): H = x.shape[-1] // 2; return torch.complex(x[..., :H], x[..., H:])
    def split(y): return torch.cat([y.real, y.imag], -1)
    N1, N2, N3 = 16, 8, 4   # 1-D length, 2-D side, 3-D side
    F1r, F1i = dft(N1, False); I1r, I1i = dft(N1, True)
    F2r, F2i = dft(N2, False); I2r, I2i = dft(N2, True)
    F3r, F3i = dft(N3, False); I3r, I3i = dft(N3, True)
    fr = torch.fft
    # 1-D, along the last dim
    rcase('fft', lambda s, x: x @ s.B, lambda s, x: split(fr.fft(x)), R(4, N1), B=breal(F1r, F1i))
    rcase('ifft', lambda s, x: x @ s.B, lambda s, x: split(fr.ifft(cplx(x))), R(4, 2 * N1), B=bcomplex(I1r, I1i))
    rcase('rfft', lambda s, x: x @ s.B, lambda s, x: split(fr.rfft(x)), R(4, N1), B=rfft_mat(N1))
    rcase('irfft', lambda s, x: x @ s.B, lambda s, x: fr.irfft(cplx(x), N1), R(4, 2 * (N1 // 2 + 1)), B=irfft_mat(N1))
    rcase('hfft', lambda s, x: (x * s.m) @ s.B, lambda s, x: fr.hfft(cplx(x), N1), R(4, 2 * (N1 // 2 + 1)), m=conjmask(N1 // 2 + 1), B=irfft_mat(N1) * N1)
    rcase('ihfft', lambda s, x: (x @ s.B) * s.m, lambda s, x: split(fr.ihfft(x)), R(4, N1), B=rfft_mat(N1) / N1, m=conjmask(N1 // 2 + 1))
    # 2-D, dims (-2, -1): the last dim first for the forward transforms, last for the inverse ones
    rcase('fft2', lambda s, x: along(x @ s.B, -2, s.Fr, s.Fi, s.P), lambda s, x: split(fr.fft2(x)), R(2, N2, N2), B=breal(F2r, F2i), Fr=F2r, Fi=F2i, P=swapP(N2))
    rcase('ifft2', lambda s, x: along(x @ s.B, -2, s.Fr, s.Fi, s.P), lambda s, x: split(fr.ifft2(cplx(x))), R(2, N2, 2 * N2), B=bcomplex(I2r, I2i), Fr=I2r, Fi=I2i, P=swapP(N2))
    rcase('rfft2', lambda s, x: along(x @ s.B, -2, s.Fr, s.Fi, s.P), lambda s, x: split(fr.rfft2(x)), R(2, N2, N2), B=rfft_mat(N2), Fr=F2r, Fi=F2i, P=swapP(N2 // 2 + 1))
    rcase('irfft2', lambda s, x: along(x, -2, s.Fr, s.Fi, s.P) @ s.B, lambda s, x: fr.irfft2(cplx(x), s=(N2, N2)), R(2, N2, 2 * (N2 // 2 + 1)), B=irfft_mat(N2), Fr=I2r, Fi=I2i, P=swapP(N2 // 2 + 1))
    rcase('hfft2', lambda s, x: along(x * s.m, -2, s.Fr, s.Fi, s.P) @ s.B, lambda s, x: fr.hfft2(cplx(x), s=(N2, N2)), R(2, N2, 2 * (N2 // 2 + 1)),
          m=conjmask(N2 // 2 + 1), B=irfft_mat(N2) * (N2 * N2), Fr=I2r, Fi=I2i, P=swapP(N2 // 2 + 1))
    rcase('ihfft2', lambda s, x: along(x @ s.B, -2, s.Fr, s.Fi, s.P) * s.m, lambda s, x: split(fr.ihfft2(x)), R(2, N2, N2),
          B=rfft_mat(N2) / (N2 * N2), Fr=F2r, Fi=F2i, P=swapP(N2 // 2 + 1), m=conjmask(N2 // 2 + 1))
    # n-D (3-D, all dims of a 4x4x4 tensor)
    def fwd3(s, x, B): y = x @ B; y = along(y, -2, s.Fr, s.Fi, s.P); return along(y, -3, s.Fr, s.Fi, s.P)
    def inv3(s, x, B): y = along(x, -3, s.Fr, s.Fi, s.P); y = along(y, -2, s.Fr, s.Fi, s.P); return y @ B
    rcase('fftn', lambda s, x: fwd3(s, x, s.B), lambda s, x: split(fr.fftn(x)), R(N3, N3, N3), B=breal(F3r, F3i), Fr=F3r, Fi=F3i, P=swapP(N3))
    rcase('ifftn', lambda s, x: inv3(s, x, s.B), lambda s, x: split(fr.ifftn(cplx(x))), R(N3, N3, 2 * N3), B=bcomplex(I3r, I3i), Fr=I3r, Fi=I3i, P=swapP(N3))
    rcase('rfftn', lambda s, x: fwd3(s, x, s.B), lambda s, x: split(fr.rfftn(x)), R(N3, N3, N3), B=rfft_mat(N3), Fr=F3r, Fi=F3i, P=swapP(N3 // 2 + 1))
    rcase('irfftn', lambda s, x: inv3(s, x, s.B), lambda s, x: fr.irfftn(cplx(x), s=(N3, N3, N3)), R(N3, N3, 2 * (N3 // 2 + 1)), B=irfft_mat(N3), Fr=I3r, Fi=I3i, P=swapP(N3 // 2 + 1))
    rcase('hfftn', lambda s, x: inv3(s, x * s.m, s.B), lambda s, x: fr.hfftn(cplx(x), s=(N3, N3, N3)), R(N3, N3, 2 * (N3 // 2 + 1)),
          m=conjmask(N3 // 2 + 1), B=irfft_mat(N3) * N3 ** 3, Fr=I3r, Fi=I3i, P=swapP(N3 // 2 + 1))
    rcase('ihfftn', lambda s, x: fwd3(s, x, s.B) * s.m, lambda s, x: split(fr.ihfftn(x)), R(N3, N3, N3),
          B=rfft_mat(N3) / N3 ** 3, Fr=F3r, Fi=F3i, P=swapP(N3 // 2 + 1), m=conjmask(N3 // 2 + 1))
    # helpers: the frequency vectors are constants (added to the input so the case has one; the shifts
    # are permutations of every dim: index_select with constant index vectors, 5 x 7 so that fftshift
    # and ifftshift differ)
    # (aten.fft_fftfreq / rfftfreq have no torch-mlir lowering: the exported graph adds the vector as a buffer)
    rcase('fftfreq', lambda s, x: x + s.f, lambda s, x: x + fr.fftfreq(N1), R(N1), f=fr.fftfreq(N1))
    rcase('rfftfreq', lambda s, x: x + s.f, lambda s, x: x + fr.rfftfreq(N1), R(N1 // 2 + 1), f=fr.rfftfreq(N1))
    def roll_idx(n, k): return (torch.arange(n) - k) % n            # index_select(i) = x[(i - k) mod n] = roll by k
    rcase('fftshift', lambda s, x: x.index_select(0, s.i0).index_select(1, s.i1), lambda s, x: fr.fftshift(x), R(5, 7), i0=roll_idx(5, 2), i1=roll_idx(7, 3))
    rcase('ifftshift', lambda s, x: x.index_select(0, s.i0).index_select(1, s.i1), lambda s, x: fr.ifftshift(x), R(5, 7), i0=roll_idx(5, -2), i1=roll_idx(7, -3))
    if fn == '--list': return sorted(C)
    if fn not in C: raise SystemExit('unknown function ' + fn)
    return C[fn]()

if __name__ == '__main__':
    if sys.argv[1] == '--list': print('\n'.join(build('--list'))); sys.exit(0)
    fn, mlir_out, bin_out = sys.argv[1:4]
    m, x = build(fn); m = m.eval()
    with torch.no_grad(): y = m.reference(x) if hasattr(m, 'reference') else m(x)
    with torch.no_grad(): assert torch.allclose(m(x), y, atol=1e-4, rtol=1e-4), 'exported body != reference'   # float32 DFT sums
    print('%s: in %s -> out %s' % (fn, list(x.shape), list(y.shape)))
    mod = fx.export_and_import(m, x, output_type='linalg-on-tensors', func_name='net')
    open(mlir_out, 'w').write(str(mod))
    xf = np.ascontiguousarray(x.numpy()).astype(np.float32).ravel(); yf = np.ascontiguousarray(y.numpy()).astype(np.float32).ravel()
    with open(bin_out, 'wb') as f:
        f.write(struct.pack('i', xf.size)); f.write(xf.tobytes()); f.write(struct.pack('i', yf.size)); f.write(yf.tobytes())
