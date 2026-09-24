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
    if fn == '--list': return sorted(C)
    if fn not in C: raise SystemExit('unknown function ' + fn)
    return C[fn]()

if __name__ == '__main__':
    if sys.argv[1] == '--list': print('\n'.join(build('--list'))); sys.exit(0)
    fn, mlir_out, bin_out = sys.argv[1:4]
    m, x = build(fn); m = m.eval()
    with torch.no_grad(): y = m.reference(x) if hasattr(m, 'reference') else m(x)
    with torch.no_grad(): assert torch.allclose(m(x), y, atol=1e-5), 'exported body != reference'
    print('%s: in %s -> out %s' % (fn, list(x.shape), list(y.shape)))
    mod = fx.export_and_import(m, x, output_type='linalg-on-tensors', func_name='net')
    open(mlir_out, 'w').write(str(mod))
    xf = np.ascontiguousarray(x.numpy()).astype(np.float32).ravel(); yf = np.ascontiguousarray(y.numpy()).astype(np.float32).ravel()
    with open(bin_out, 'wb') as f:
        f.write(struct.pack('i', xf.size)); f.write(xf.tobytes()); f.write(struct.pack('i', yf.size)); f.write(yf.tobytes())
