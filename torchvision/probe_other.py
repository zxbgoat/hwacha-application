#!/usr/bin/env python3
"""Feasibility probe for the non-classification torchvision families: build the model, wrap it so the
forward returns one float tensor, run PyTorch, then torch-mlir export to linalg-on-tensors.
usage: probe_other.py <family> <name>   -> prints one line: OK/FAIL <stage> <detail>"""
import sys, traceback, warnings, torch, torchvision.models as M
warnings.simplefilter('ignore')
fam, name = sys.argv[1:3]
torch.manual_seed(0)
class W(torch.nn.Module):
    def __init__(s, m, f): super().__init__(); s.m = m; s.f = f
    def forward(s, x): return s.f(s.m, x)
def build():
    if fam == 'quantization':
        m = M.get_model(name, weights=None, quantize=True); return W(m, lambda m, x: m(x)), torch.randn(1,3,32,32)
    if fam == 'segmentation':
        m = M.get_model(name, weights=None, weights_backbone=None, aux_loss=False); return W(m, lambda m, x: m(x)['out']), torch.randn(1,3,32,32)
    if fam == 'detection':
        kw = {} if 'ssd' in name else dict(min_size=64, max_size=64)
        m = M.get_model(name, weights=None, weights_backbone=None, **kw)
        # scores of the detections (variable count): export will tell us whether the graph is static
        return W(m, lambda m, x: m([x[0]])[0]['scores']), torch.randn(1,3,64,64)
    if fam == 'video':
        m = M.get_model(name, weights=None)
        if name.startswith('mvit'): x = torch.randn(1,3,16,224,224)
        elif name.startswith('swin3d'): x = torch.randn(1,3,16,32,32)
        else: x = torch.randn(1,3,8,32,32)
        return W(m, lambda m, x: m(x)), x
    if fam == 'optical_flow':
        m = M.get_model(name, weights=None)
        return W(m, lambda m, x: m(x[0:1], x[1:2], num_flow_updates=2)[-1]), torch.randn(2,3,64,64)
stage = 'build'
try:
    m, x = build(); m.eval()
    stage = 'forward'
    with torch.no_grad(): y = m(x)
    stage = 'export'
    from torch_mlir import fx
    mod = fx.export_and_import(m, x, output_type='linalg-on-tensors', func_name='net')
    txt = str(mod); ops = sorted(set(t.split()[0] for t in txt.split('\n') if t.strip().startswith(('linalg.', 'tensor.', 'tm_tensor.', 'torch.'))))
    print(f"OK {name} in={list(x.shape)} out={list(y.shape)} params={sum(p.numel() for p in m.parameters())/1e6:.1f}M ir={len(txt)>>20}MB ops={','.join(o for o in ops if not o.startswith('linalg.') or o in ('linalg.generic','linalg.conv_3d_ncdhw_fcdhw'))}")
except Exception as e:
    msg = str(e).strip().split('\n'); msg = next((l for l in msg if 'error' in l.lower() or 'Error' in l), msg[0])
    print(f"FAIL {name} at {stage}: {msg[:200]}")
