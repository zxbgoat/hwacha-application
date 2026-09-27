#!/usr/bin/env python3
"""Export one torch_geometric.nn module / function (pytorch-geometric.readthedocs.io/en/latest/modules/nn.html,
torch_geometric 2.8) through torch-mlir to linalg-on-tensors, plus a PyTorch reference for one call.
Every case runs on one small fixed graph: 8 nodes with 4 features, 16 directed edges (self-loop free,
one undirected graph for the layers that need it), 2 graphs in the batch vector; edge indices, edge
attributes, positions, batch vectors and such are constant buffers, the node features are the input.
Integer / bool outputs are cast to float, several outputs are flattened and concatenated. Modules run
in eval mode with random weights (fixed seed).    usage: export_tg.py <name> <mlir_out> <check_out>"""
import sys, os, math, struct, numpy as np, torch, torch.nn as nn, torch.nn.functional as NF
import torch_geometric.nn as G, torch_geometric.utils as U, torch_geometric.utils.loop as LOOP
import tg_lib as LB
# The self-loop utilities filter the edge list with a boolean mask (edge_index[:, src != dst]): a
# data-dependent shape that torch.export keeps symbolic and torch-mlir's importer then crashes on.
# The case graph has no self loops, so on it remove_self_loops is the identity and add_self_loops /
# add_remaining_self_loops are the concatenation of the constant loop index; the static versions below
# do exactly that. Every module binds the names at import time, so they are swapped in every loaded
# torch_geometric module (and back for the reference, see with_orig).
_orig = dict(remove_self_loops=LOOP.remove_self_loops, add_self_loops=LOOP.add_self_loops, add_remaining_self_loops=LOOP.add_remaining_self_loops)
def _static_remove(edge_index, edge_attr=None): return edge_index, edge_attr
def _static_add(edge_index, edge_attr=None, fill_value=None, num_nodes=None):
    N = num_nodes if isinstance(num_nodes, int) else int(edge_index.max()) + 1 if edge_index.numel() else 0
    loop = torch.arange(N, device=edge_index.device).view(1, -1).repeat(2, 1)
    full = torch.cat([edge_index, loop], 1)
    if edge_attr is not None:
        la = LOOP.compute_loop_attr(edge_index, edge_attr, N, False, fill_value)
        edge_attr = torch.cat([edge_attr, la], 0)
    return full, edge_attr
# scatter (every aggregation of every layer funnels through torch_geometric.utils.scatter) exports as
# tm_tensor.scatter, which hwacha-mlir does not take; the static version is the one-hot matmul
# onehot(index).T @ src (sum / mean), or a masked max / min over the broadcast (n, E, F) block. degree
# is the column sum of the same one-hot. The reference runs with PyG's own.
import torch_geometric.utils._scatter as SC, torch_geometric.utils._degree as DG
_orig.update(scatter=SC.scatter, degree=DG.degree)
def _static_scatter(src, index, dim=0, dim_size=None, reduce='sum'):
    assert index.dim() == 1
    if src.numel() == 0 and dim_size is not None: return _orig['scatter'](src, index, dim, dim_size, reduce)   # empty (TGN's flush)
    if dim < 0: dim += src.dim()
    n = dim_size if dim_size is not None else int(index.max()) + 1   # (Tensor.max is the table lookup while exporting)
    if dim != 0: return _static_scatter(src.transpose(0, dim), index, 0, n, reduce).transpose(0, dim)
    sh = src.shape; src2 = src.reshape(sh[0], int(np.prod(sh[1:])))
    o = (index[:, None] == torch.arange(n, device=src.device)[None, :]).to(src.dtype)   # (E, n)
    if reduce in ('sum', 'add'): out = o.t() @ src2
    elif reduce == 'mean': out = (o.t() @ src2) / o.sum(0).clamp(min=1)[:, None]
    elif reduce in ('max', 'amax', 'min', 'amin'):
        sgn = 1.0 if 'max' in reduce else -1.0; big = torch.finfo(src.dtype).max if src.is_floating_point() else torch.iinfo(src.dtype).max
        blk = torch.where(o.t()[:, :, None] > 0, sgn * src2[None], torch.full_like(src2[None].expand(n, -1, -1), -big)).amax(1)
        out = torch.where(o.sum(0)[:, None] > 0, sgn * blk, torch.zeros_like(blk))
    elif reduce == 'mul': out = torch.exp(o.t() @ torch.log(src2.abs().clamp(min=1e-30))) * torch.cos(math.pi * (o.t() @ (src2 < 0).to(src.dtype)))
    else: raise ValueError(reduce)
    return out.reshape((n,) + tuple(sh[1:]))
def _static_degree(index, num_nodes=None, dtype=None):
    n = num_nodes if num_nodes is not None else int(index.max()) + 1
    return (index[:, None] == torch.arange(n, device=index.device)[None, :]).sum(0).to(dtype or torch.long)
# The neighbour searches of pyg-lib (torch.ops.pyg.knn / radius) have no fake-tensor kernel. Every
# search in a case runs on constant positions (a buffer), so its result is a constant too: the static
# versions compute it with tg_lib's tensor algorithms on the real values recorded for the buffer
# (keyed by shape, dtype and the call's arguments) and hand it over as a constant.
# During the eager reference run every search result is recorded in call order; the export then
# replays them from buffers registered on the case (a tensor created inside the trace, however
# constant, gets an unbacked size), so the searches are graph constants with static shapes.
_search_log = []; _replay = None
def _search(f):
    def g(*a, **k):
        global _replay
        if _replay is not None: t = _replay.pop(0); return t
        r = f(*a, **k); _search_log.append(r); return r
    return g
def _static_knn(x, y, k, batch_x=None, batch_y=None, cosine=False, num_workers=1, batch_size=None): return LB.knn_pairs(x, y, k, batch_x, batch_y)
def _static_knn_graph(x, k, batch=None, loop=False, flow='source_to_target', cosine=False, num_workers=1, batch_size=None):
    e = LB.knn_graph(x, k, loop, batch); return e if flow == 'source_to_target' else e.flip(0)
def _static_radius(x, y, r, batch_x=None, batch_y=None, max_num_neighbors=32, num_workers=1, batch_size=None): return LB.radius_pairs(x, y, r, True, batch_x, batch_y)
def _static_radius_graph(x, r, batch=None, loop=False, max_num_neighbors=32, flow='source_to_target', num_workers=1, batch_size=None):
    p = LB.radius_pairs(x, x, r, loop, batch, batch); e = torch.stack([p[1], p[0]]); return e if flow == 'source_to_target' else e.flip(0)
_static_knn, _static_knn_graph, _static_radius, _static_radius_graph = map(_search, (_static_knn, _static_knn_graph, _static_radius, _static_radius_graph))
# DimeNet's triplets (an index computation through torch_sparse on the constant edge list): recorded and replayed too
import torch_geometric.nn.models.dimenet as DMN
_orig['triplets'] = DMN.triplets
def _tup_search(f):
    def g(*a, **k):
        global _replay
        if _replay is not None: n = _replay.pop(0).item(); return tuple(_replay.pop(0) for _ in range(n))
        r = f(*a, **k); _search_log.append(torch.tensor(len(r))); _search_log.extend(r); return r
    return g
_static_triplets = _tup_search(DMN.triplets)
def record_searches(m, x):
    """eager forward with the static searches recording; their results become buffers replayed by the export"""
    global _replay
    _search_log.clear(); _replay = None
    with torch.no_grad(): m(x)
    for i, t in enumerate(_search_log): m.register_buffer('search_%d' % i, t.clone())
    for t in _search_log:   # the searched indices' maxima (scatter defaults over them)
        if not t.is_floating_point() and t.dim() >= 1 and t.numel():
            for r in (t.view(-1),) + (tuple(t) if t.dim() == 2 else ()): _maxtab[(tuple(r.shape), r.dtype)] = max(_maxtab.get((tuple(r.shape), r.dtype), -1), int(r.max()))
    _replay = [getattr(m, 'search_%d' % i) for i in range(len(_search_log))]
_orig.update(knn=G.knn, knn_graph=G.knn_graph, radius=G.radius, radius_graph=G.radius_graph)
# to_dense_batch (the aggregations that work on padded sets: LSTM / GRU / Sort / MLP / LCM / SetTransformer / GMT /
# PatchTransformer, and MemPooling / SGFormer) fills the padded (B, max, F) block with out[idx] = x, an index_put
# (tm_tensor.scatter), after a cumsum (tm_tensor.scan). The static version builds the placement as a one-hot
# (B*max, N) matrix from the index (constant): position of node n = n - first node of its graph, computed
# with the one-hot cumulative count (a triangular matmul instead of cumsum).
import torch_geometric.utils._to_dense_batch as TDB
_orig['to_dense_batch'] = TDB.to_dense_batch
def _static_to_dense_batch(x, batch=None, fill_value=0.0, max_num_nodes=None, batch_size=None):
    Nn = x.shape[0]
    if batch is None: return x.unsqueeze(0), torch.ones(1, Nn, dtype=torch.bool)
    B = batch_size if batch_size is not None else int(batch.max()) + 1
    o = (batch[:, None] == torch.arange(B)[None, :]).to(x.dtype)               # (N, B)
    cnt = o.sum(0)                                                              # (B,)
    Mx = max_num_nodes if max_num_nodes is not None else int(cnt.long().max())   # (an int tensor: the export's max lookup)
    tri = torch.tril(torch.ones(Nn, Nn, dtype=x.dtype), -1)                     # tri[n, m] = 1 for m < n
    pos = (tri * (o @ o.t())).sum(1)                                            # nodes of the same graph before n
    slot = (o * torch.arange(B, dtype=x.dtype)[None, :]).sum(1) * Mx + pos      # (N,) flat slot
    P = (slot[:, None] == torch.arange(B * Mx, dtype=x.dtype)[None, :]).to(x.dtype)   # (N, B*Mx)
    out = (P.t() @ x.reshape(Nn, -1)).reshape(B, Mx, *x.shape[1:])
    mask = P.sum(0).reshape(B, Mx) > 0
    if not (isinstance(fill_value, (int, float)) and fill_value == 0.0): out = out + (~mask).to(x.dtype).reshape(B, Mx, *([1] * (x.dim() - 1))) * fill_value
    return out, mask
import torch_geometric.nn.aggr.fused as FUS, torch_geometric.nn.aggr.multi as MUL
# MultiAggregation (also inside DegreeScalerAggregation) fuses sum / mean / max / .. into FusedAggregation,
# whose degree count is a scatter_add_ (no lowering); FusedAggregation.forward is replaced by running the
# fused aggregations one by one (each through the static scatter) -- the same values.
import torch_geometric.nn.aggr.basic as AGBASIC
def _fused_forward(self, x, index=None, ptr=None, dim_size=None, dim=-2):
    return [getattr(AGBASIC, n)()(x, index, ptr, dim_size, dim) for n in self.aggr_names]   # (the fusable aggregations are parameter free)
FUS.FusedAggregation.forward = _fused_forward
_static = dict(to_dense_batch=_static_to_dense_batch, remove_self_loops=_static_remove, add_self_loops=_static_add, add_remaining_self_loops=_static_add, scatter=_static_scatter, degree=_static_degree,
               knn=_static_knn, knn_graph=_static_knn_graph, radius=_static_radius, radius_graph=_static_radius_graph, triplets=_static_triplets)
def swap_loops(table):
    """bind the given versions of the utilities in every loaded torch_geometric module (they import the names)"""
    for mod in [m for n, m in list(sys.modules.items()) if n.startswith('torch_geometric') and m is not None]:
        for k, v in list(vars(mod).items()):
            kk = 'add_self_loops' if k == 'add_self_loops_fn' else k
            if kk in table and (v in _orig.values() or v in _static.values()): setattr(mod, k, table[kk])
swap_loops(_static)
# torch.atan2 (the bond angles of DimeNet / the point-pair features of PPFConv) calls libm's atan2f, which
# hwacha-cc has no vector version of; while exporting it is torch.atan of the ratio with the quadrant fixes
# (as tafunc/taf_lib.py's). Here b >= 0 (a norm), so the angle is in [0, pi].
_torch_atan2 = torch.atan2
def _static_atan2(y, x):
    ax = torch.where(x == 0, torch.full_like(x, 1e-30), x); a = torch.atan(y / ax)
    return torch.where(x < 0, a + torch.where(y >= 0, torch.full_like(a, math.pi), torch.full_like(a, -math.pi)), a)
# int(index.max()) + 1 (the default dim_size / batch_size / num_classes / num_nodes of a dozen PyG helpers)
# is a data-dependent integer torch.export cannot resolve. Every index in a case is a constant, so
# `.max()` of an integer tensor is looked up from a table of the real values recorded when the case is
# built (keyed by shape and dtype; an ambiguous key falls back to the largest, and the per-case
# `sizes` note where that matters). Active only while exporting.
from torch._subclasses.fake_tensor import FakeTensor
_maxtab = {}
def record_maxes(m):
    for _, t in list(m.named_buffers()):
        if not t.is_floating_point() and t.dtype != torch.bool and t.numel(): _maxtab[(tuple(t.shape), t.dtype)] = max(_maxtab.get((tuple(t.shape), t.dtype), -1), int(t.max()))
        if not t.is_floating_point() and t.dtype != torch.bool and t.dim() == 1 and t.numel():          # counts per value (to_dense_batch / PatchTransformer): (n_values,) int64
            cnt = torch.bincount(t)
            for n_ in range(cnt.numel(), 2 * N + 1): _maxtab[((n_,), torch.int64)] = max(_maxtab.get(((n_,), torch.int64), -1), int(cnt.max()))
        if not t.is_floating_point() and t.dim() == 2 and t.shape[0] == 2:   # edge_index rows
            for r in range(2): _maxtab[((t.shape[1],), t.dtype)] = max(_maxtab.get(((t.shape[1],), t.dtype), -1), int(t[r].max()))
            _maxtab[((t.numel(),), t.dtype)] = max(_maxtab.get(((t.numel(),), t.dtype), -1), int(t.max()))
_tmax = torch.Tensor.max
import torch_geometric.nn.aggr.base as AGB
AGB.Aggregation.assert_sorted_index = lambda self, index: None   # the sorted-index cases hand over a sorted constant
_exporting = False
def _fake_max(t, *a, **k):
    if _exporting and not t.is_floating_point() and not a and not k:
        key = (tuple(t.shape), t.dtype)
        if key in _maxtab: return torch.tensor(_maxtab[key], dtype=t.dtype)
    return _tmax(t, *a, **k)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch_mlir import fx
import inspect, torch_mlir.extras.fx_importer as _fxi
# torch.export constant-folds the index buffers of aten.index (x[idx] with idx a buffer) into a literal
# torch.Tensor inside the op's list argument, which torch-mlir's importer only accepts as a graph Node
# ("Heterogeneous lists are not supported"). Import a tensor operand as a vtensor literal instead.
import textwrap
_src = textwrap.dedent(inspect.getsource(_fxi.GraphNodeImporter._import_list_argument))
_old = """        else:
            assert (is_optional_type and operand_type is NoneType) or ("""
assert _old in _src
_new = """        elif isinstance(operand, torch.Tensor):
            with loc:
                val = self._import_literal(operand)
        else:
            assert (is_optional_type and operand_type is NoneType) or ("""
_ns = dict(vars(_fxi)); exec("from typing import *\n" + _src.replace(_old, _new), _ns)
_fxi.GraphNodeImporter._import_list_argument = _ns['_import_list_argument']

class Case(nn.Module):
    def __init__(s, body, mods=None, **bufs):
        super().__init__(); s.body = body
        for k, v in (mods() if callable(mods) else (mods or {})).items(): setattr(s, k, v)
        for k, v in bufs.items(): s.register_buffer(k, v)
        for m in s.modules():   # non-persistent buffers must be in the state dict for torch-mlir's importer
            for n in list(m._non_persistent_buffers_set): t = getattr(m, n); delattr(m, n); m.register_buffer(n, t)
    def forward(s, x): return s.body(s, x)
class Ref(Case):
    """exported body differs from the reference (the genuine torch_geometric call, what check.bin holds)"""
    def __init__(s, body, ref, mods=None, _atol=1e-4, **bufs): super().__init__(body, mods, **bufs); s.ref = ref; s.atol = _atol
    def reference(s, x): return s.ref(s, x)
def export_linalg(m, x):
    from torch_mlir.extras.fx_decomp_util import get_decomposition_table
    table = {k: v for k, v in get_decomposition_table().items() if not any(str(k).startswith('aten.' + n) for n in ('zeros', 'ones', 'full', 'new_zeros', 'new_ones', 'new_full', 'empty_like', 'zeros_like', 'ones_like', 'full_like'))}
    return fx.export_and_import(torch.export.export(m, (x,)), x, output_type='linalg-on-tensors', func_name='net', decomposition_table=table)
def R(*shape): return torch.randn(*shape)
def flat(*ts): return torch.cat([t.reshape(-1).float() for t in ts])

# ---- the graph: 8 nodes, 4 features, 16 directed edges without self loops (8 undirected pairs), 2 graphs
N, F, E = 8, 4, 16
def graph():
    torch.manual_seed(0)
    pairs = torch.tensor([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [3, 4], [4, 5], [4, 6], [5, 6], [5, 7], [6, 7], [2, 4], [0, 3], [1, 4], [3, 6], [4, 7]])[:8]
    ei = torch.cat([pairs.t(), pairs.flip(1).t()], 1)                 # (2, 16) undirected
    ea = torch.rand(E, 3)                                             # edge features
    ew = torch.rand(E)                                                # edge weights
    pos = torch.randn(N, 3)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    return ei, ea, ew, pos, batch
EI, EA, EW, POS, BATCH = graph()

def build(name):
    torch.manual_seed(0)
    C = {}
    def with_orig(f):
        """run f with PyG's genuine self-loop utilities (the reference)"""
        def g(s, x):
            global _replay
            swap_loops(_orig); rp = _replay; _replay = None
            try: return f(s, x)
            finally: swap_loops(_static); _replay = rp
        return g
    def case(n, body, x=None, mods=None, **b): C[n] = lambda: (Case(body, mods, ei=EI, **b), R(N, F) if x is None else x)
    def rcase(n, body, ref, x=None, mods=None, **b): C[n] = lambda: (Ref(body, with_orig(ref), mods, ei=EI, **b), R(N, F) if x is None else x)
    def L(n, mk, f=lambda s, x: s.m(x, s.ei), **b): rcase(n, f, f, None, lambda: dict(m=mk()), **b)
    O = 6
    class VEnc(nn.Module):   # variational encoder: (mu, logstd)
        def __init__(s): super().__init__(); s.a = G.GCNConv(F, O); s.b = G.GCNConv(F, O)
        def forward(s, x, ei): return s.a(x, ei), s.b(x, ei)
    # ---- convolutional layers
    L('simple_conv', lambda: G.SimpleConv(aggr='mean'))
    L('gcn_conv', lambda: G.GCNConv(F, O))
    def cheb(s, x):   # T0 = x, T1 = L~ x, Tk = 2 L~ T(k-1) - T(k-2); lambda_max = 2 (the default)
        Tx0 = x; out = s.m.lins[0](Tx0); Tx1 = s.Lt @ x; out = out + s.m.lins[1](Tx1)
        for lin in s.m.lins[2:]: Tx2 = 2 * (s.Lt @ Tx1) - Tx0; out = out + lin(Tx2); Tx0, Tx1 = Tx1, Tx2
        return out + s.m.bias
    rcase('cheb_conv', cheb, lambda s, x: s.m(x, s.ei, s.ew), None, lambda: dict(m=G.ChebConv(F, O, K=3)), ew=EW, Lt=LB.cheb_norm(EI, N, EW))
    L('sage_conv', lambda: G.SAGEConv(F, O))
    L('graph_conv', lambda: G.GraphConv(F, O))
    L('gated_graph_conv', lambda: G.GatedGraphConv(O, num_layers=2))
    L('res_gated_graph_conv', lambda: G.ResGatedGraphConv(F, O))
    L('gat_conv', lambda: G.GATConv(F, O, heads=2))
    L('gatv2_conv', lambda: G.GATv2Conv(F, O, heads=2))
    L('transformer_conv', lambda: G.TransformerConv(F, O, heads=2))
    L('agnn_conv', lambda: G.AGNNConv())
    L('tag_conv', lambda: G.TAGConv(F, O, K=2))
    L('gin_conv', lambda: G.GINConv(nn.Sequential(nn.Linear(F, O), nn.ReLU(), nn.Linear(O, O))))
    L('gine_conv', lambda: G.GINEConv(nn.Sequential(nn.Linear(F, O), nn.ReLU(), nn.Linear(O, O)), edge_dim=3), lambda s, x: s.m(x, s.ei, s.ea), ea=EA)
    L('arma_conv', lambda: G.ARMAConv(F, O, num_stacks=2, num_layers=2))
    L('sg_conv', lambda: G.SGConv(F, O, K=2))
    L('ssg_conv', lambda: G.SSGConv(F, O, alpha=0.1, K=2))
    L('appnp', lambda: G.APPNP(K=3, alpha=0.1))
    import torch_geometric.nn.conv.mf_conv as MFC
    def mf_forward(self, x, edge_index):   # MFConv.forward with the per-degree linears picked by the one-hot of the degree (index_copy_ has no lowering)
        if isinstance(x, torch.Tensor): x = (x, x)
        N_ = x[0].shape[0]; deg = LB.onehot(LB.adj(edge_index, N_).sum(1).clamp(max=self.max_degree).long() if False else torch.zeros(1).long(), 1) if False else None
        A = (edge_index[1][:, None] == torch.arange(N_)[None, :]).float().t() @ (edge_index[0][:, None] == torch.arange(N_)[None, :]).float()   # A[dst, src]
        d = A.sum(1).clamp(max=self.max_degree); h = A @ x[0]; out = 0
        for i, (ll, lr) in enumerate(zip(self.lins_l, self.lins_r)): out = out + (d == i).float()[:, None] * (ll(h) + lr(x[1]))
        return out
    MFC.MFConv.forward = mf_forward
    def mf(s, x):   # MFConv: h = sum_j x_j, out_i = lin_l[deg_i](h_i) + lin_r[deg_i](x_i) with deg clamped to max_degree: every degree's linear evaluated, picked by the one-hot of deg
        m = s.m; h = s.A @ x; out = 0
        for i, (ll, lr) in enumerate(zip(m.lins_l, m.lins_r)): out = out + s.degoh[:, i:i + 1] * (ll(h) + lr(x))
        return out
    deg = LB.adj(EI, N).sum(1).clamp(max=10).long()
    rcase('mf_conv', mf, lambda s, x: s.m(x, s.ei), None, lambda: dict(m=G.MFConv(F, O)), A=LB.adj(EI, N), degoh=LB.onehot(deg, 11))
    et = torch.arange(E) % 2; ei_r = [EI[:, et == r] for r in range(2)]
    def rgcn(s, x):   # sum over relations of mean-aggregated W_r x_j (num_bases None, aggr 'mean'), + root weight + bias
        out = x @ s.m.root + s.m.bias
        for r in range(2): out = out + s.m.comp_r(s.Ar[r] @ x, r) if hasattr(s.m, 'comp_r') else out + (s.Ar[r] @ x) @ s.m.weight[r]
        return out
    rcase('rgcn_conv', rgcn, lambda s, x: s.m(x, s.ei, s.et), None, lambda: dict(m=G.RGCNConv(F, O, num_relations=2)), et=et, Ar=torch.stack([LB.adj(e, N) / LB.adj(e, N).sum(1, keepdim=True).clamp(min=1) for e in ei_r]))
    rcase('fast_rgcn_conv', rgcn, lambda s, x: s.m(x, s.ei, s.et), None, lambda: dict(m=G.FastRGCNConv(F, O, num_relations=2)), et=et, Ar=torch.stack([LB.adj(e, N) / LB.adj(e, N).sum(1, keepdim=True).clamp(min=1) for e in ei_r]))
    L('rgat_conv', lambda: G.RGATConv(F, O, num_relations=2), lambda s, x: s.m(x, s.ei, s.et), et=torch.arange(E) % 2)
    L('signed_conv', lambda: G.SignedConv(F, O, first_aggr=True), lambda s, x: s.m(x, s.ei, s.ei.flip(0)))
    L('dna_conv', lambda: G.DNAConv(F, heads=2, groups=2), lambda s, x: s.m(x.unsqueeze(1).expand(-1, 2, -1), s.ei))
    L('point_net_conv', lambda: G.PointNetConv(nn.Linear(F + 3, O), nn.Linear(O, O)), lambda s, x: s.m(x, s.pos, s.ei), pos=POS)
    L('gmm_conv', lambda: G.GMMConv(F, O, dim=3, kernel_size=2), lambda s, x: s.m(x, s.ei, s.ea), ea=EA)
    def spline(s, x):   # message_j = sum_k basis_jk * (x_j @ W[widx_jk]); mean aggregation; + root weight + bias
        m = s.m; W = m.weight                                                     # (kernel_size**D, F, O)
        xj = x[s.ei[0]]                                                            # (E, F)
        msg = sum(s.basis[:, k, None] * torch.einsum('ef,efo->eo', xj, W[s.widx[:, k]]) for k in range(s.basis.shape[1]))
        out = LB.scatter_mean(msg, s.ei[1], N)
        return out + x @ m.lin.weight.t() + m.bias
    basis, widx = LB.spline_basis(EA, 2)
    rcase('spline_conv', spline, lambda s, x: s.m(x, s.ei, s.ea), None, lambda: dict(m=G.SplineConv(F, O, dim=3, kernel_size=2)), ea=EA, basis=basis, widx=widx)
    L('nn_conv', lambda: G.NNConv(F, O, nn.Linear(3, F * O)), lambda s, x: s.m(x, s.ei, s.ea), ea=EA)
    L('cg_conv', lambda: G.CGConv(F, dim=3), lambda s, x: s.m(x, s.ei, s.ea), ea=EA)
    L('edge_conv', lambda: G.EdgeConv(nn.Linear(2 * F, O)))
    def dyn_edge(s, x):   # pyg's knn of the input features (the point itself is among its k nearest; pinned to the fixed input as a constant), then EdgeConv with max aggregation
        src = s.knn.reshape(-1); dst = torch.arange(N).repeat_interleave(3)
        msg = s.m.nn(torch.cat([x[dst], x[src] - x[dst]], 1))
        return LB.scatter_max(msg, dst, N)
    xin = R(N, F)
    rcase('dynamic_edge_conv', dyn_edge, lambda s, x: s.m(x), xin, lambda: dict(m=G.DynamicEdgeConv(nn.Linear(2 * F, O), k=3)), knn=LB.knn_idx(xin, 3, loop=True))
    def xconv(s, x):   # XConv.forward with pyg's knn (K nearest of pos, the point itself included) replaced by the constant neighbour lists
        m = s.m; K = m.kernel_size; col = s.knn.reshape(-1); row = torch.arange(N).repeat_interleave(K)
        rel = s.pos[col] - s.pos[row]                                             # (N*K, D)
        x_star = m.mlp1(rel)                                                       # (N, K, C_delta)
        x_star = torch.cat([x_star, x[col].view(N, K, F)], -1).transpose(1, 2).contiguous()
        tm = m.mlp2(rel.view(N, K * 3))                                            # (N, K, K)
        return m.conv(torch.matmul(x_star, tm))
    rcase('x_conv', xconv, lambda s, x: s.m(x, s.pos), None, lambda: dict(m=G.XConv(F, O, dim=3, kernel_size=3)), pos=POS, knn=LB.knn_idx(POS, 3, loop=True))
    L('ppf_conv', lambda: G.PPFConv(nn.Linear(F + 4, O), nn.Linear(O, O)), lambda s, x: s.m(x, s.pos, s.nrm, s.ei), pos=POS, nrm=NF.normalize(R(N, 3), dim=1))
    L('feast_conv', lambda: G.FeaStConv(F, O, heads=2))
    L('point_transformer_conv', lambda: G.PointTransformerConv(F, O), lambda s, x: s.m(x, s.pos, s.ei), pos=POS)
    def hyper(s, x):   # HypergraphConv (no attention): out = D^-1 H B^-1 H^T (x Theta) + bias, H the (N, M) incidence, D node degrees, B hyperedge sizes (1/0 -> 0)
        m = s.m; H = s.H; xt = m.lin(x)
        D = 1.0 / H.sum(1); D = torch.where(torch.isinf(D), torch.zeros_like(D), D); Bd = 1.0 / H.sum(0); Bd = torch.where(torch.isinf(Bd), torch.zeros_like(Bd), Bd)
        out = H.t() @ xt * Bd[:, None]; out = H @ out * D[:, None]
        return out + m.bias
    hi = torch.stack([torch.arange(N), torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])])
    rcase('hypergraph_conv', hyper, lambda s, x: s.m(x, s.hi, num_edges=4), None, lambda: dict(m=G.HypergraphConv(F, O)), hi=hi, H=LB.onehot(hi[1], 4))
    L('le_conv', lambda: G.LEConv(F, O))
    L('pna_conv', lambda: G.PNAConv(F, O, aggregators=['mean', 'max', 'min', 'std'], scalers=['identity', 'amplification', 'attenuation'], deg=torch.tensor([0, 2, 4, 2]), towers=1))
    def cluster_gcn(s, x):   # A~ = A + I with weights deg_inv[col] (+ diag_lambda * deg_inv on the diagonal), out = lin_out(A~ x) + lin_root(x)
        return s.m.lin_out(s.At @ x) + s.m.lin_root(x)
    def cluster_gcn_A():
        A = LB.adj(EI, N) + torch.eye(N); deg_inv = 1.0 / A.sum(1).clamp(min=1)          # A[dst, src]: deg over dst (col in PyG's naming)
        return deg_inv[:, None] * A + torch.diag(0.2 * deg_inv)
    rcase('cluster_gcn_conv', cluster_gcn, lambda s, x: s.m(x, s.ei), None, lambda: dict(m=G.ClusterGCNConv(F, O, diag_lambda=0.2)), At=cluster_gcn_A())
    L('gen_conv', lambda: G.GENConv(F, O, num_layers=2))
    L('gcn2_conv', lambda: G.GCN2Conv(F, alpha=0.1, theta=0.5, layer=1), lambda s, x: s.m(x, x, s.ei))
    def pan(s, x):   # panentropy: M = sum_{i<=L} (prod_{j<=i} w_j) A^i (A = 1s adjacency, A^0 = I; mul_nnz accumulates the weights),
        m = s.m; P = torch.eye(N); c = m.weight[0]; M = c * P   # normalised D^-1/2 M D^-1/2 with D = the row count of M's non-zeros; out = lin(M x)
        for i in range(1, m.filter_size + 1): P = P @ s.A; c = c * m.weight[i]; M = M + c * P
        dis = s.nnz.sum(1).pow(-0.5)
        return m.lin((dis[:, None] * M * dis[None, :]) @ x)
    def pan_nnz():
        A = LB.adj(EI, N); P = torch.eye(N); S = P.clone()
        for i in range(1, 3): P = P @ A; S = S + P
        return (S > 0).float()
    rcase('pan_conv', pan, lambda s, x: s.m(x, s.ei)[0], None, lambda: dict(m=G.PANConv(F, O, filter_size=2)), A=LB.adj(EI, N), nnz=pan_nnz())
    L('wl_conv_continuous', lambda: G.WLConvContinuous())
    def film(s, x):   # per relation r: (beta, gamma) = film_r(x_i), message = act(gamma * lin_r(x_j) + beta), mean aggregation; plus the skip relation on x_i
        m = s.m; out = torch.zeros(N, O)
        for r in range(2):
            bg = m.films[r](x); beta, gamma = bg[:, :O], bg[:, O:]; h = m.lins[r](x)
            msg = m.act(gamma[s.ei_r[r][1]] * h[s.ei_r[r][0]] + beta[s.ei_r[r][1]])
            out = out + LB.scatter_mean(msg, s.ei_r[r][1], N)
        bg = m.film_skip(x); beta, gamma = bg[:, :O], bg[:, O:]
        return out + m.act(gamma * m.lin_skip(x) + beta)
    rcase('film_conv', film, lambda s, x: s.m(x, s.ei, s.et), None, lambda: dict(m=G.FiLMConv(F, O, num_relations=2)), et=et, ei_r=torch.stack(ei_r))
    L('super_gat_conv', lambda: G.SuperGATConv(F, O, heads=2))
    L('fa_conv', lambda: G.FAConv(F), lambda s, x: s.m(x, x, s.ei))
    L('eg_conv', lambda: G.EGConv(F, O, num_heads=2, num_bases=2))
    L('pdn_conv', lambda: G.PDNConv(F, O, edge_dim=3, hidden_channels=8), lambda s, x: s.m(x, s.ei, s.ea), ea=EA)
    L('general_conv', lambda: G.GeneralConv(F, O, in_edge_channels=3, attention=True), lambda s, x: s.m(x, s.ei, s.ea), ea=EA)
    L('lg_conv', lambda: G.LGConv())
    L('point_gnn_conv', lambda: G.PointGNNConv(nn.Linear(F, 3), nn.Linear(F + 3, F), nn.Linear(F, F)), lambda s, x: s.m(x, s.pos, s.ei), pos=POS)
    def gps(s, x):   # GPSConv.forward with to_dense_batch given batch_size / max_num_nodes (its int(batch.max()) breaks the export)
        m = s.m; h = m.conv(x, s.ei); h = NF.dropout(h, m.dropout, training=False); h = h + x
        if m.norm1 is not None: h = m.norm1(h, batch=s.batch) if m.norm_with_batch else m.norm1(h)
        hd = x.view(2, 4, F)     # to_dense_batch on 2 graphs of 4 nodes each: a reshape, the mask all true (a[mask] = the reshape back)
        a, _ = m.attn(hd, hd, hd, need_weights=False); a = a.reshape(N, F)
        a = NF.dropout(a, m.dropout, training=False); a = a + x
        if m.norm2 is not None: a = m.norm2(a, batch=s.batch) if m.norm_with_batch else m.norm2(a)
        out = h + a; out = out + m.mlp(out)
        if m.norm3 is not None: out = m.norm3(out, batch=s.batch) if m.norm_with_batch else m.norm3(out)
        return out
    rcase('gps_conv', gps, lambda s, x: s.m(x, s.ei, s.batch), None, lambda: dict(m=G.GPSConv(F, G.GCNConv(F, F), heads=2, attn_type='multihead')), batch=BATCH)
    L('anti_symmetric_conv', lambda: G.AntiSymmetricConv(F, num_iters=2))
    L('dir_gnn_conv', lambda: G.DirGNNConv(G.GCNConv(F, O)))
    L('mix_hop_conv', lambda: G.MixHopConv(F, O, powers=[0, 1, 2]))
    # ---- aggregation operators: x (E, F) = the messages of the 16 edges aggregated to their target nodes
    def A(n, mk, f=lambda s, x: s.m(x, s.idx, dim_size=N), x=None, **b): rcase(n, f, f, R(E, F) if x is None else x, lambda: dict(m=mk()), idx=EI[1], **b)
    A('sum_aggregation', lambda: G.SumAggregation())
    A('mean_aggregation', lambda: G.MeanAggregation())
    A('max_aggregation', lambda: G.MaxAggregation())
    A('min_aggregation', lambda: G.MinAggregation())
    A('mul_aggregation', lambda: G.MulAggregation())
    A('var_aggregation', lambda: G.VarAggregation())
    A('std_aggregation', lambda: G.StdAggregation())
    A('softmax_aggregation', lambda: G.SoftmaxAggregation(t=0.5))
    A('power_mean_aggregation', lambda: G.PowerMeanAggregation(p=2.0), x=R(E, F).abs())
    A('multi_aggregation', lambda: G.MultiAggregation(['mean', 'max', 'min'], mode='cat'))
    A('multi_aggregation_proj', lambda: G.MultiAggregation(['mean', 'max'], mode='proj', mode_kwargs=dict(in_channels=F, out_channels=O)))
    rcase('median_aggregation', lambda s, x: LB.quantile_static(x, s.idx, N, 0.5, 'lower'), lambda s, x: s.m(x, s.idx, dim_size=N), R(E, F), lambda: dict(m=G.MedianAggregation()), idx=EI[1])
    rcase('quantile_aggregation', lambda s, x: LB.quantile_static(x, s.idx, N, 0.25, 'linear'), lambda s, x: s.m(x, s.idx, dim_size=N), R(E, F), lambda: dict(m=G.QuantileAggregation(q=0.25)), idx=EI[1])
    A('lstm_aggregation', lambda: G.LSTMAggregation(F, O), lambda s, x: s.m(x, s.idx_sorted, dim_size=N), idx_sorted=EI[1].sort().values)
    A('gru_aggregation', lambda: G.GRUAggregation(F, O), lambda s, x: s.m(x, s.idx_sorted, dim_size=N), idx_sorted=EI[1].sort().values)
    import torch_geometric.nn.aggr.set2set as S2S
    def s2s_forward(self, x, index=None, ptr=None, dim_size=None, dim=-2):   # Set2Set.forward; the LSTM state built with torch.zeros (the LSTM's returned state comes back 4-D in torch.export's trace)
        h = (torch.zeros(self.lstm.num_layers, dim_size, x.size(-1)), torch.zeros(self.lstm.num_layers, dim_size, x.size(-1)))
        q_star = torch.zeros(dim_size, self.out_channels)
        for _ in range(self.processing_steps):
            q, (hn, cn) = self.lstm(q_star.unsqueeze(0), h); q = q.view(dim_size, self.in_channels)
            h = (hn.view(self.lstm.num_layers, dim_size, -1), cn.view(self.lstm.num_layers, dim_size, -1))   # the state re-shaped: torch.export otherwise returns it 4-D into the next step
            e = (x * q[index]).sum(dim=-1, keepdim=True); a = U.softmax(e, index, ptr, dim_size, dim)
            r = self.reduce(a * x, index, ptr, dim_size, dim, reduce='sum'); q_star = torch.cat([q, r], dim=-1)
        return q_star
    S2S.Set2Set.forward = s2s_forward
    A('set2set', lambda: G.Set2Set(F, processing_steps=2))
    A('degree_scaler_aggregation', lambda: G.DegreeScalerAggregation(['mean', 'max'], ['identity', 'amplification', 'attenuation'], deg=torch.tensor([0, 2, 4, 2])))
    import torch_geometric.nn.aggr.sort as SRT
    def sort_forward(self, x, index=None, ptr=None, dim_size=None, dim=-2, max_num_elements=None):   # SortAggregation.forward: the per-graph sort by the last feature as a rank one-hot (descending; padded rows, filled with min - 1, rank last)
        fill_value = x.detach().min() - 1
        bx, _ = self.to_dense_batch(x, index, ptr, dim_size, dim, fill_value=fill_value, max_num_elements=max_num_elements)
        B, Nn, D = bx.size(); key = bx[:, :, -1]; ar = torch.arange(Nn)
        gt = (key[:, None, :] > key[:, :, None]) | ((key[:, None, :] == key[:, :, None]) & (ar[None, None, :] < ar[None, :, None]))   # [b, j, i]: i before j
        rank = gt.to(x.dtype).sum(-1)                                                                            # (B, Nn)
        P = (rank[:, None, :] == torch.arange(self.k, dtype=x.dtype)[None, :, None]).to(x.dtype)                 # (B, k, Nn)
        bx = P @ bx                                                                                              # (B, k, D), rows of rank >= Nn... (k <= Nn here)
        bx = torch.where(bx == fill_value, torch.zeros_like(bx), bx)
        return bx.view(B, self.k * D)
    SRT.SortAggregation.forward = sort_forward
    A('sort_aggregation', lambda: G.SortAggregation(k=3), lambda s, x: s.m(x, s.idx_sorted, dim_size=N), idx_sorted=EI[1].sort().values)
    A('graph_multiset_transformer', lambda: G.GraphMultisetTransformer(F, k=2, heads=2), lambda s, x: s.m(x, s.idx_sorted, dim_size=N), idx_sorted=EI[1].sort().values)
    A('attentional_aggregation', lambda: G.AttentionalAggregation(nn.Linear(F, 1), nn.Linear(F, O)))
    A('mlp_aggregation', lambda: G.MLPAggregation(F, O, max_num_elements=4, num_layers=1), lambda s, x: s.m(x, s.idx_sorted, dim_size=N), idx_sorted=EI[1].sort().values)
    A('deep_sets_aggregation', lambda: G.DeepSetsAggregation(nn.Linear(F, O), nn.Linear(O, O)))
    import torch_geometric.nn.aggr.utils as AGU
    def mab_forward(self, x, y, x_mask=None, y_mask=None):   # MultiheadAttentionBlock.forward with the key-padding mask as an additive float mask and out[~x_mask] = 0 as a multiply (masked_fill_ has no lowering)
        am = None
        if y_mask is not None: am = (~y_mask).to(x.dtype)[:, None, :].expand(-1, x.shape[1], -1).repeat_interleave(self.heads, 0) * -1e9   # (B*heads, Lq, Lk)
        out, _ = self.attn(x, y, y, attn_mask=am, need_weights=False)
        if y_mask is not None:   # a query whose keys are all padding gets NaN attention in PyG (nan_to_num -> 0 at the end); here its attention output is zeroed
            out = out * y_mask.any(1).to(out.dtype)[:, None, None]
        if x_mask is not None: out = out * x_mask.to(out.dtype)[:, :, None]
        out = out + x
        if self.layer_norm1 is not None: out = self.layer_norm1(out)
        out = out + self.lin(out).relu()
        if self.layer_norm2 is not None: out = self.layer_norm2(out)
        return out
    AGU.MultiheadAttentionBlock.forward = mab_forward
    A('set_transformer_aggregation', lambda: G.SetTransformerAggregation(F, heads=2), lambda s, x: s.m(x, s.idx_sorted, dim_size=N), idx_sorted=EI[1].sort().values)
    A('lcm_aggregation', lambda: G.LCMAggregation(F, O), lambda s, x: s.m(x, s.idx_sorted, dim_size=N), idx_sorted=EI[1].sort().values)
    A('variance_preserving_aggregation', lambda: G.VariancePreservingAggregation())
    A('patch_transformer_aggregation', lambda: G.PatchTransformerAggregation(F, O, patch_size=2, hidden_channels=8, heads=2), lambda s, x: s.m(x, s.idx_sorted, dim_size=N), idx_sorted=EI[1].sort().values)
    # ---- attention (dense, on a (1, N, C) sequence)
    case('performer_attention', lambda s, x: s.m(x.unsqueeze(0)).squeeze(0), None, lambda: dict(m=G.PerformerAttention(channels=F, heads=2, head_channels=4)))
    case('qformer', lambda s, x: s.m(x.unsqueeze(0)).squeeze(0), None, lambda: dict(m=G.QFormer(input_dim=F, hidden_dim=8, output_dim=O, num_heads=2, num_layers=1)))
    import torch_geometric.nn.attention.sgformer as SGA
    def sg_attn_forward(self, x, mask=None):   # SGFormerAttention.forward with qs[qs == 0] = eps written as torch.where (a masked assignment lowers to tm_tensor.scan)
        B, Nn, _ = x.shape; qs = self.q(x).reshape(B, Nn, self.heads, self.head_channels); ks = self.k(x).reshape(B, Nn, self.heads, self.head_channels); vs = self.v(x).reshape(B, Nn, self.heads, self.head_channels)
        eps = 1e-6; qs = torch.where(qs == 0, torch.full_like(qs, eps), qs); ks = torch.where(ks == 0, torch.full_like(ks, eps), ks)
        qs = qs / torch.linalg.norm(qs, ord=2, dim=-1, keepdim=True); ks = ks / torch.linalg.norm(ks, ord=2, dim=-1, keepdim=True)
        kvs = torch.einsum("blhm,blhd->bhmd", ks, vs); num = torch.einsum("bnhm,bhmd->bnhd", qs, kvs) + Nn * vs
        ks_sum = ks.sum(1); den = torch.einsum("bnhm,bhm->bnh", qs, ks_sum).unsqueeze(-1) + Nn
        return (num / den).mean(dim=2)
    SGA.SGFormerAttention.forward = sg_attn_forward
    case('sgformer_attention', lambda s, x: s.m(x.unsqueeze(0)).squeeze(0), None, lambda: dict(m=G.SGFormerAttention(channels=F, heads=2, head_channels=4)))
    case('polynormer_attention', lambda s, x: s.m(x.unsqueeze(0)).squeeze(0), None, lambda: dict(m=G.PolynormerAttention(channels=F, heads=2, head_channels=4)))
    # ---- normalization layers
    def NM(n, mk, f=lambda s, x: s.m(x), **b): rcase(n, f, f, None, lambda: dict(m=mk()), batch=BATCH, **b)
    NM('batch_norm', lambda: G.BatchNorm(F))
    NM('hetero_batch_norm', lambda: G.HeteroBatchNorm(F, num_types=2), lambda s, x: s.m(x, s.batch))
    NM('instance_norm', lambda: G.InstanceNorm(F, affine=True), lambda s, x: s.m(x, s.batch, batch_size=2))
    NM('layer_norm', lambda: G.LayerNorm(F, mode='graph'), lambda s, x: s.m(x, s.batch, batch_size=2))
    NM('layer_norm_node', lambda: G.LayerNorm(F, mode='node'))
    NM('hetero_layer_norm', lambda: G.HeteroLayerNorm(F, num_types=2), lambda s, x: s.m(x, s.batch))
    NM('graph_norm', lambda: G.GraphNorm(F), lambda s, x: s.m(x, s.batch, batch_size=2))
    NM('graph_size_norm', lambda: G.GraphSizeNorm(), lambda s, x: s.m(x, s.batch, batch_size=2))
    NM('pair_norm', lambda: G.PairNorm(), lambda s, x: s.m(x, s.batch))
    NM('mean_subtraction_norm', lambda: G.MeanSubtractionNorm(), lambda s, x: s.m(x, s.batch, dim_size=2))
    NM('message_norm', lambda: G.MessageNorm(learn_scale=True), lambda s, x: s.m(x, s.msg), msg=R(N, F))
    NM('diff_group_norm', lambda: G.DiffGroupNorm(F, groups=2))
    # ---- pooling layers (global pools: (N, F) with the batch vector -> (2, F))
    case('global_add_pool', lambda s, x: G.global_add_pool(x, s.batch, size=2), None, batch=BATCH)
    case('global_mean_pool', lambda s, x: G.global_mean_pool(x, s.batch, size=2), None, batch=BATCH)
    case('global_max_pool', lambda s, x: G.global_max_pool(x, s.batch, size=2), None, batch=BATCH)
    def PL(n, mk, f, **b): rcase(n, f, f, None, lambda: dict(m=mk()), batch=BATCH, **b)
    def topk_pool(s, x):   # SelectTopK: score = tanh(x . w / |w|), the top ratio*4 = 2 per graph; pooled x = x[perm] * score[perm] (rows graph by graph)
        w = s.m.select.weight; score = torch.tanh((x * w).sum(-1) / w.norm())
        P = LB.topk_perm_batched(score, s.batch, [2, 2])
        return (P @ x) * (P @ score).view(-1, 1)
    rcase('topk_pooling', topk_pool, lambda s, x: s.m(x, s.ei, batch=s.batch)[0], None, lambda: dict(m=G.TopKPooling(F, ratio=0.5)), batch=BATCH)
    def sag_pool(s, x):   # SAGPooling: attn = GraphConv(x) (1 channel), then SelectTopK as above
        a = s.m.gnn(x, s.ei); w = s.m.select.weight; score = torch.tanh((a * w).sum(-1) / w.norm())
        P = LB.topk_perm_batched(score, s.batch, [2, 2])
        return (P @ x) * (P @ score).view(-1, 1)
    rcase('sag_pooling', sag_pool, lambda s, x: s.m(x, s.ei, batch=s.batch)[0], None, lambda: dict(m=G.SAGPooling(F, ratio=0.5)), batch=BATCH)
    # EdgePooling / ClusterPooling: the edge contraction is a greedy matching (resp. connected components) in
    # Python over the sorted edge scores -- data dependent by nature. With the case's scores the matching is a
    # constant: the cluster assignment is precomputed from an eager run (the reference) and the export does
    # the score computation and the pooled sum / scaling with that constant cluster matrix.
    def edge_pool(s, x):   # e = lin([x_i, x_j]) -> softmax over target -> + 0.5; new_x = C^T x scaled by the merged edge's score (1 for singletons)
        m = s.m; e = m.lin(torch.cat([x[s.ei[0]], x[s.ei[1]]], -1)).view(-1)
        e = LB.scatter_softmax(e.view(-1, 1), s.ei[1], N).view(-1) + m.add_to_edge_score
        sc = s.esel @ e + s.single                                            # (n_clusters,): the contracted edge's score, or 1
        return (s.C.t() @ x) * sc.view(-1, 1)
    class EdgePoolCase(Ref):
        """the cluster constants come from the module's own eager forward on the case input"""
        def __init__(s, cls):
            super().__init__(edge_pool if cls is G.EdgePooling else cluster_pool, lambda s, x: s.m(x, s.ei, s.batch)[0], lambda: dict(m=cls(F)), ei=EI, batch=BATCH)
            x = R(N, F); s.x = x
            with torch.no_grad(): swap_loops(_orig); _, _, _, info = s.m(x, EI, BATCH); swap_loops(_static)
            cluster = info.cluster; n = int(cluster.max()) + 1; s.register_buffer('C', LB.onehot(cluster, n))
            if cls is G.ClusterPooling:   # the nodes on no contracted edge (threshold 0.5 on tanh scores... the module's default threshold)
                with torch.no_grad(): swap_loops(_orig); e = torch.tanh(s.m.lin(torch.cat([x[EI[0]], x[EI[1]]], -1)).view(-1)); swap_loops(_static)
                ec = EI[:, e > s.m.threshold]; deg = torch.zeros(N).index_put((ec[0],), torch.ones(ec.shape[1]), accumulate=True) + torch.zeros(N).index_put((ec[1],), torch.ones(ec.shape[1]), accumulate=True)
                s.register_buffer('single', (deg == 0).float()); s.register_buffer('Esrc', LB.onehot(EI[0], N)); s.register_buffer('Edst', LB.onehot(EI[1], N))
            if cls is G.EdgePooling:
                esel = torch.zeros(n, E); single = torch.ones(n)
                with torch.no_grad(): sc = info.new_edge_score
                # the clusters are numbered in matching order: cluster i < number of contracted edges got edge new_edge_indices[i]
                with torch.no_grad(): swap_loops(_orig); e = s.m.lin(torch.cat([x[EI[0]], x[EI[1]]], -1)).view(-1); e = U.softmax(e, EI[1], num_nodes=N) + s.m.add_to_edge_score; swap_loops(_static)
                perm = torch.argsort(e, descending=True).tolist(); mask = torch.ones(N, dtype=torch.bool); i = 0
                for idx in perm:
                    a_, b_ = int(EI[0, idx]), int(EI[1, idx])
                    if not mask[a_] or not mask[b_]: continue
                    esel[i, idx] = 1.0; single[i] = 0.0; mask[a_] = False; mask[b_] = False; i += 1
                s.register_buffer('esel', esel); s.register_buffer('single', single)
    C['edge_pooling'] = lambda: (lambda c: (c, c.x))(EdgePoolCase(G.EdgePooling))
    def cluster_pool(s, x):   # ClusterPooling: e = tanh(lin([x_i, x_j])); clusters = connected components of the edges with e > threshold (a constant here);
        m = s.m; e = torch.tanh(m.lin(torch.cat([x[s.ei[0]], x[s.ei[1]]], -1)).view(-1))   # x_out = (S C)^T x with S the dense score matrix (S_ii = 1 on single nodes)
        S = s.Esrc.t() @ (e[:, None] * s.Edst) + torch.diag(s.single)                 # S[src, dst] = e (one-hot products instead of index_put)
        return (S @ s.C).t() @ x
    C['cluster_pooling'] = lambda: (lambda c: (c, c.x))(EdgePoolCase(G.ClusterPooling))
    def asa_pool(s, x):   # ASAPooling.forward up to the pooled x: self loops added, x_q = lin(max_j x_j), att over [x_q_i, x_j] with softmax per target, x = sum_j score x_j, fitness = sigmoid(LEConv(x)), top-k, x[perm] * fitness[perm]
        m = s.m; ei = s.ei_sl; src, dst = ei[0], ei[1]
        x_pool_j = x[src]; x_q = LB.scatter_max(x_pool_j, dst, N); x_q = m.lin(x_q)[dst]
        score = m.att(torch.cat([x_q, x_pool_j], -1)).view(-1); score = NF.leaky_relu(score, m.negative_slope)
        score = LB.scatter_softmax(score.view(-1, 1), dst, N).view(-1)
        xs = LB.scatter_sum(x[src] * score.view(-1, 1), dst, N)
        fit = m.gnn_score(xs, ei).sigmoid().view(-1)
        w = m.select.weight; P = LB.topk_perm_batched(torch.tanh(fit * w.view(-1) / w.norm()), s.batch, [2, 2])   # SelectTopK ranks tanh(fit * w / |w|)
        return (P @ xs) * (P @ fit).view(-1, 1)
    rcase('asa_pooling', asa_pool, lambda s, x: s.m(x, s.ei, batch=s.batch)[0], None, lambda: dict(m=G.ASAPooling(F, ratio=0.5)), batch=BATCH, ei_sl=torch.cat([EI, torch.arange(N).repeat(2, 1)], 1))
    def pan_pool(s, x):   # PANPooling.forward with M dense: s = beta0 * (x . p) + beta1 * colsum(M); SelectTopK: score = tanh(s * w / |w|), top-k (4 of 8), x[perm] * score[perm]
        m = s.m; sc = m.beta[0] * (x * m.p).sum(-1) + m.beta[1] * s.M.sum(0)
        w = m.select.weight; score = torch.tanh(sc * w.view(-1) / w.norm())
        P = LB.topk_perm_static(score, N // 2); return (P @ x) * (P @ score).view(-1, 1)
    from torch_sparse import SparseTensor as SpT
    rcase('pan_pooling', pan_pool, lambda s, x: s.m(x, SpT.from_dense(s.M))[0], None, lambda: dict(m=G.PANPooling(F, ratio=0.5)), M=LB.adj(EI, N) + torch.eye(N))
    def mem_pool(s, x):   # MemPooling.forward on the dense (2, 4, F) batch (to_dense_batch = a reshape here), cdist written out
        m = s.m; B, Nn = 2, 4; xd = x.view(B, Nn, F); H, K = m.heads, m.num_clusters
        d2 = ((m.k.view(H * K, 1, -1) - xd.view(1, B * Nn, -1)) ** 2).sum(-1)
        dist = (1. + d2 / m.tau).pow(-(m.tau + 1.0) / 2.0).view(H, K, B, Nn).permute(2, 0, 3, 1)
        S = dist / dist.sum(-1, keepdim=True); S = m.conv(S).squeeze(1).softmax(-1)
        return m.lin(S.transpose(1, 2) @ xd)
    rcase('mem_pooling', mem_pool, lambda s, x: s.m(x, s.batch)[0], None, lambda: dict(m=G.MemPooling(F, O, heads=2, num_clusters=2)), batch=BATCH)
    case('max_pool_x', lambda s, x: G.max_pool_x(s.cluster, x, s.batch, size=2)[0], None, batch=BATCH, cluster=torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]))
    case('avg_pool_x', lambda s, x: G.avg_pool_x(s.cluster, x, s.batch, size=2)[0], None, batch=BATCH, cluster=torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]))
    from torch_geometric.data import Data
    rcase('max_pool', lambda s, x: G.max_pool_x(s.cluster, x, s.batch, size=2)[0], lambda s, x: G.max_pool(s.cluster, Data(x=x, edge_index=s.ei, batch=s.batch)).x, None, batch=BATCH, cluster=torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]))
    rcase('avg_pool', lambda s, x: G.avg_pool_x(s.cluster, x, s.batch, size=2)[0], lambda s, x: G.avg_pool(s.cluster, Data(x=x, edge_index=s.ei, batch=s.batch)).x, None, batch=BATCH, cluster=torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]))
    case('max_pool_neighbor_x', lambda s, x: G.max_pool_neighbor_x(Data(x=x, edge_index=s.ei)).x, None)
    case('avg_pool_neighbor_x', lambda s, x: G.avg_pool_neighbor_x(Data(x=x, edge_index=s.ei)).x, None)
    # graclus: torch_cluster's kernel visits the nodes in a random permutation (not reproducible: the same
    # call gives different matchings); the case is the greedy heavy-edge matching in index order, as a
    # tensor computation on the weighted adjacency (one node per step, its best unmatched neighbour)
    case('graclus', lambda s, x: LB.graclus_t(s.A, N).float(), torch.zeros(1), A=LB.adj(EI, N, EW))
    rcase('voxel_grid', lambda s, x: LB.voxel_grid(x, 1.0).float(), lambda s, x: G.voxel_grid(x, size=1.0).float(), R(N, 3))
    rcase('fps', lambda s, x: LB.fps_t(x, 4).float(), lambda s, x: G.fps(x, ratio=0.5, random_start=False).float(), R(N, 3))
    rcase('knn', lambda s, x: LB.knn_mask(x, s.y, 2), lambda s, x: LB.pair_mask(G.knn(x, s.y, k=2), 3, N), R(N, 3), y=R(3, 3))   # the (3, N) 0/1 mask of the 2 nearest of each query
    rcase('knn_graph', lambda s, x: LB.knn_mask(x, x, 2, loop=False), lambda s, x: LB.pair_mask(G.knn_graph(x, k=2).flip(0), N, N), R(N, 3))   # the (N, N) 0/1 mask [target, source]
    rcase('radius', lambda s, x: LB.radius_mask(x, s.y, 1.5).float(), lambda s, x: LB.adj(G.radius(x, s.y, r=1.5).flip(0), max(N, 3))[:3, :N].float() if False else LB.pair_mask(G.radius(x, s.y, r=1.5), 3, N).float(), R(N, 3), y=R(3, 3))
    rcase('radius_graph', lambda s, x: LB.radius_mask(x, x, 1.5, loop=False).float(), lambda s, x: LB.pair_mask(G.radius_graph(x, r=1.5).flip(0), N, N).float(), R(N, 3))
    rcase('nearest', lambda s, x: LB.nearest(x, s.y).float(), lambda s, x: G.nearest(x, s.y).float(), R(N, 3), y=R(3, 3))
    # ---- unpooling
    rcase('knn_interpolate', lambda s, x: G.knn_interpolate(x, s.px, s.py, k=3), lambda s, x: G.knn_interpolate(x, s.px, s.py, k=3), R(N, F), px=POS, py=R(5, 3))
    # ---- models
    def M(n, mk, f=lambda s, x: s.m(x, s.ei), x=None, **b): rcase(n, f, f, x, lambda: dict(m=mk()), batch=BATCH, **b)
    M('mlp', lambda: G.MLP([F, 8, O]), lambda s, x: s.m(x))
    M('gcn', lambda: G.GCN(F, 8, num_layers=2, out_channels=O))
    M('graph_sage', lambda: G.GraphSAGE(F, 8, num_layers=2, out_channels=O))
    M('gin', lambda: G.GIN(F, 8, num_layers=2, out_channels=O))
    M('gat', lambda: G.GAT(F, 8, num_layers=2, out_channels=O, heads=2))
    M('pna', lambda: G.PNA(F, 8, num_layers=2, out_channels=O, aggregators=['mean', 'max'], scalers=['identity', 'amplification'], deg=torch.tensor([0, 2, 4, 2])))
    M('edge_cnn', lambda: G.EdgeCNN(F, 8, num_layers=2, out_channels=O))
    M('jumping_knowledge', lambda: G.JumpingKnowledge('lstm', channels=F, num_layers=2), lambda s, x: s.m([x, x * 2, x.relu()]))
    M('jumping_knowledge_max', lambda: G.JumpingKnowledge('max'), lambda s, x: s.m([x, x * 2, x.relu()]))
    M('hetero_jumping_knowledge', lambda: G.HeteroJumpingKnowledge(['a', 'b'], 'lstm', channels=F, num_layers=2), lambda s, x: flat(*s.m({'a': [x, x * 2], 'b': [x.relu(), -x]}).values()))
    class EdgeM(nn.Module):
        def __init__(s): super().__init__(); s.l = nn.Linear(2 * F + 3, 3)
        def forward(s, src, dst, ea, u, batch): return s.l(torch.cat([src, dst, ea], 1))
    class NodeM(nn.Module):
        def __init__(s): super().__init__(); s.l = nn.Linear(F + 3, O)
        def forward(s, x, ei, ea, u, batch): return s.l(torch.cat([x, LB.scatter_sum(ea, ei[1], N)], 1))
    class GlobM(nn.Module):
        def __init__(s): super().__init__(); s.l = nn.Linear(2 + O, 2)
        def forward(s, x, ei, ea, u, batch): return s.l(torch.cat([u, LB.scatter_mean(x, batch, 2)], 1))
    M('meta_layer', lambda: G.MetaLayer(EdgeM(), NodeM(), GlobM()), lambda s, x: flat(*s.m(x, s.ei, s.ea, s.u, s.batch)), ea=EA, u=R(2, 2))
    M('node2vec', lambda: G.Node2Vec(EI, embedding_dim=O, walk_length=3, context_size=2, num_nodes=N), lambda s, x: s.m(s.nodes), x=torch.zeros(1), nodes=torch.arange(N))
    M('deep_graph_infomax', lambda: G.DeepGraphInfomax(hidden_channels=8, encoder=G.GCNConv(F, 8), summary=lambda z, *a, **k: torch.sigmoid(z.mean(0)), corruption=lambda x, ei: (x.flip(0), ei)), lambda s, x: flat(*s.m(x, s.ei)))
    M('inner_product_decoder', lambda: G.InnerProductDecoder(), lambda s, x: s.m(x, s.ei))
    M('gae', lambda: G.GAE(G.GCNConv(F, O)), lambda s, x: s.m.decoder(s.m.encode(x, s.ei), s.ei))
    M('vgae', lambda: G.VGAE(VEnc()), lambda s, x: s.m.decoder(s.m.encode(x, s.ei), s.ei))
    M('arga', lambda: G.ARGA(G.GCNConv(F, O), nn.Linear(O, 1)), lambda s, x: flat(s.m.decoder(s.m.encode(x, s.ei), s.ei), s.m.discriminator(s.m.encode(x, s.ei))))
    M('argva', lambda: G.ARGVA(VEnc(), nn.Linear(O, 1)), lambda s, x: flat(s.m.decoder(s.m.encode(x, s.ei), s.ei), s.m.discriminator(s.m.encode(x, s.ei))))
    M('signed_gcn', lambda: G.SignedGCN(F, O, num_layers=2), lambda s, x: s.m(x, s.ei[:, :8], s.ei[:, 8:]))
    from torch_geometric.data import Data as DataC
    renet_fields = dict(sub=torch.tensor([0, 1]), rel=torch.tensor([0, 1]), obj=torch.tensor([2, 3]), h_sub=torch.tensor([4, 5, 6, 7]), h_obj=torch.tensor([1, 2, 3, 0]), h_sub_t=torch.tensor([0, 1, 0, 1]), h_obj_t=torch.tensor([0, 1, 0, 1]), h_sub_batch=torch.tensor([0, 0, 1, 1]), h_obj_batch=torch.tensor([0, 0, 1, 1]))
    M('renet', lambda: G.RENet(num_nodes=N, num_rels=2, hidden_channels=8, seq_len=2), lambda s, x: flat(*s.m(DataC(**{k: getattr(s, 'rn_' + k) for k in renet_fields}))), x=torch.zeros(1), **{'rn_' + k: v for k, v in renet_fields.items()})
    def graph_unet(s, x):   # GraphUNet depth 1 on the dense adjacency: GCN(improved) -> A2 = (A+I)^2 without the diagonal -> TopKPooling (4 of 8) -> GCN -> unpool (P^T) + skip -> GCN
        m = s.m; A = s.A
        x = m.act(LB.gcn_dense(m.down_convs[0], x, A, True)); res = x
        A2 = (A + torch.eye(N)) @ (A + torch.eye(N)); A2 = A2 * (1 - torch.eye(N))
        pool = m.pools[0]; w = pool.select.weight; score = torch.tanh((x * w).sum(-1) / w.norm())
        P = LB.topk_perm_static(score, N // 2)                                   # (4, N)
        xp = (P @ x) * (P @ score).view(-1, 1); Ap = P @ A2 @ P.t()
        xp = m.act(LB.gcn_dense(m.down_convs[1], xp, Ap, True))
        up = P.t() @ xp
        return LB.gcn_dense(m.up_convs[0], res + up, A, True)
    rcase('graph_unet', graph_unet, lambda s, x: s.m(x, s.ei), None, lambda: dict(m=G.GraphUNet(F, 8, O, depth=1, pool_ratios=0.5)), A=LB.adj(EI, N))
    M('schnet', lambda: G.SchNet(hidden_channels=8, num_filters=8, num_interactions=1, num_gaussians=5, cutoff=5.0), lambda s, x: s.m(s.z, s.pos), x=torch.zeros(1), z=torch.tensor([1, 6, 8, 1, 6, 8, 1, 6]), pos=POS)
    M('dimenet', lambda: G.DimeNet(hidden_channels=8, out_channels=1, num_blocks=1, num_bilinear=2, num_spherical=3, num_radial=3, cutoff=5.0), lambda s, x: s.m(s.z, s.pos), x=torch.zeros(1), z=torch.tensor([1, 6, 8, 1, 6, 8, 1, 6]), pos=POS)
    M('dimenet_plus_plus', lambda: G.DimeNetPlusPlus(hidden_channels=8, out_channels=1, num_blocks=1, int_emb_size=4, basis_emb_size=2, out_emb_channels=4, num_spherical=3, num_radial=3, cutoff=5.0), lambda s, x: s.m(s.z, s.pos), x=torch.zeros(1), z=torch.tensor([1, 6, 8, 1, 6, 8, 1, 6]), pos=POS)
    M('metapath2vec', lambda: G.MetaPath2Vec({('a', 'to', 'b'): EI[:, :8], ('b', 'to', 'a'): EI[:, 8:]}, embedding_dim=O, metapath=[('a', 'to', 'b'), ('b', 'to', 'a')], walk_length=2, context_size=2, num_nodes_dict={'a': N, 'b': N}), lambda s, x: s.m('a', s.nodes), x=torch.zeros(1), nodes=torch.arange(N))
    M('deep_gcn_layer', lambda: G.DeepGCNLayer(G.GENConv(F, F), G.LayerNorm(F), nn.ReLU(), block='res+'))
    yoh = NF.one_hot(torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])).float(); lmask = torch.tensor([True, True, False, True, False, True, False, False])
    rcase('label_propagation', lambda s, x: s.m(s.y * s.maskf.view(-1, 1), s.ei), lambda s, x: s.m(s.y, s.ei, mask=s.mask), torch.zeros(1), lambda: dict(m=G.LabelPropagation(num_layers=2, alpha=0.9)), y=yoh, mask=lmask, maskf=lmask.float())
    def cs(s, x):   # CorrectAndSmooth.correct (autoscale) + smooth with the masked assignments as one-hot / where: error = M (y - y_soft), scale = sigma / |smoothed| (inf or > 1000 -> 1), y = y + scale * smoothed; smooth: y[mask] = y_true, propagate
        m = s.m; Mo = s.Moh; yt = s.yoh                                            # Mo (4, N) rows of the 4 labelled nodes, yt (4, 3)
        err = Mo.t() @ (yt - Mo @ x)                                               # (N, 3), zero off the mask
        sm = m.prop1(err, s.ei, post_step=lambda t: t.clamp(-1., 1.))
        sigma = (Mo @ err).abs().sum() / 4; scale = sigma / sm.abs().sum(1, keepdim=True)
        scale = torch.where(torch.isinf(scale) | (scale > 1000), torch.ones_like(scale), scale)
        y = x + scale * sm
        y = y * (1 - Mo.sum(0))[:, None] + Mo.t() @ yt                             # smooth: the labelled rows replaced by the truth
        return m.prop2(y, s.ei)
    yl = torch.tensor([0, 1, 2, 0]); ml = torch.tensor([0, 1, 3, 5])
    rcase('correct_and_smooth', cs, lambda s, x: s.m.smooth(s.m.correct(x, s.y, s.mask, s.ei), s.y, s.mask, s.ei), R(N, 3).softmax(1), lambda: dict(m=G.CorrectAndSmooth(num_correction_layers=2, correction_alpha=0.5, num_smoothing_layers=2, smoothing_alpha=0.5)), y=yl, mask=ml, Moh=LB.onehot(ml, N), yoh=NF.one_hot(yl, 3).float())
    M('attentive_fp', lambda: G.AttentiveFP(F, 8, O, edge_dim=3, num_layers=2, num_timesteps=2), lambda s, x: s.m(x, s.ei, s.ea, s.batch), ea=EA)
    M('rect_l', lambda: G.RECT_L(F, 8), lambda s, x: s.m(x, s.ei))
    M('linkx', lambda: G.LINKX(N, F, 8, O, num_layers=1))
    M('light_gcn', lambda: G.LightGCN(N, O, num_layers=2), lambda s, x: s.m(s.ei), x=torch.zeros(1))
    rcase('mask_label', lambda s, x: x + s.maskf.view(-1, 1) * s.m.emb(s.y), lambda s, x: s.m(x, s.y, s.mask), None, lambda: dict(m=G.MaskLabel(3, F)), y=torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]), mask=lmask, maskf=lmask.float())
    M('group_add_rev', lambda: G.GroupAddRev(G.GCNConv(F // 2, F // 2), num_groups=2, disable=True), lambda s, x: s.m(x, s.ei))
    M('gnnff', lambda: G.GNNFF(hidden_node_channels=8, hidden_edge_channels=8, num_layers=1, cutoff=5.0), lambda s, x: s.m(s.z, s.pos), x=torch.zeros(1), z=torch.tensor([1, 6, 8, 1, 6, 8, 1, 6]), pos=POS)
    M('pmlp', lambda: G.PMLP(F, 8, O, num_layers=2), lambda s, x: s.m(x, s.ei))
    M('neural_fingerprint', lambda: G.NeuralFingerprint(F, 8, O, num_layers=2), lambda s, x: s.m(x, s.ei, s.batch))
    # ViSNet's NeighborEmbedding drops the self loops of the (loop=True) radius graph with a boolean mask; with
    # the loops at the tail of every node's neighbour list... no: pyg's order puts the loop first per node. The
    # mask is data dependent; the export keeps the loops and zeroes their messages (W = 0 on the loop edges)
    import torch_geometric.nn.models.visnet as VSN
    def ne_forward(self, z, x, edge_index, edge_weight, edge_attr):
        C = self.cutoff(edge_weight); W = self.distance_proj(edge_attr) * C.view(-1, 1)
        W = W * (edge_index[0] != edge_index[1]).to(W.dtype).view(-1, 1)
        xn = self.embedding(z); xn = self.propagate(edge_index, x=xn, W=W)
        return self.combine(torch.cat([x, xn], dim=1))
    VSN.NeighborEmbedding.forward = ne_forward
    def dist_forward(self, pos, batch):   # edge_weight[mask] = |edge_vec|[mask]: the loop edges have a zero vector, so |edge_vec| is the same
        edge_index = VSN.radius_graph(pos, r=self.cutoff, batch=batch, loop=self.add_self_loops, max_num_neighbors=self.max_num_neighbors)
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        return edge_index, torch.norm(edge_vec, dim=-1), edge_vec
    VSN.Distance.forward = dist_forward
    def block_forward(self, z, pos, batch):   # ViSNetBlock.forward with edge_vec[mask] /= |edge_vec|[mask] as a division by max(|edge_vec|, eps) (the loop edges are zero vectors)
        x = self.embedding(z)
        edge_index, edge_weight, edge_vec = self.distance(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)
        edge_vec = edge_vec / torch.norm(edge_vec, dim=1).clamp(min=1e-30).unsqueeze(1)
        edge_vec = self.sphere(edge_vec)
        x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)
        vec = torch.zeros(x.size(0), ((self.lmax + 1) ** 2) - 1, x.size(1), dtype=x.dtype, device=x.device)
        edge_attr = self.edge_embedding(edge_index, edge_attr, x)
        for attn in self.vis_mp_layers[:-1]:
            dx, dvec, dedge_attr = attn(x, vec, edge_index, edge_weight, edge_attr, edge_vec)
            x = x + dx; vec = vec + dvec; edge_attr = edge_attr + dedge_attr
        dx, dvec, _ = self.vis_mp_layers[-1](x, vec, edge_index, edge_weight, edge_attr, edge_vec)
        x = x + dx; vec = vec + dvec
        return self.out_norm(x), self.vec_out_norm(vec)
    VSN.ViSNetBlock.forward = block_forward
    M('visnet', lambda: G.ViSNet(lmax=1, num_heads=2, num_layers=1, hidden_channels=8, num_rbf=4, cutoff=5.0, max_num_neighbors=8, derivative=False), lambda s, x: s.m(s.z, s.pos, s.batch)[0], x=torch.zeros(1), z=torch.tensor([1, 6, 8, 1, 6, 8, 1, 6]), pos=POS)
    import torch_geometric.nn.models.sgformer as SGF
    def trans_forward(self, x, batch):   # SGModule.forward with to_dense_batch = a reshape (2 graphs of 4 sorted nodes, the mask all true)
        x = x.view(2, 4, -1); x = self.fcs[0](x); x = self.bns[0](x); x = self.activation(x); layer_ = [x]
        for i, attn in enumerate(self.attns):
            x = attn(x, None); x = (x + layer_[i]) / 2.; x = self.bns[i + 1](x); x = self.activation(x); layer_.append(x)
        return x.reshape(N, -1)
    SGF.SGModule.forward = trans_forward
    M('sgformer', lambda: G.SGFormer(F, 8, O, trans_num_heads=2, trans_num_layers=1, gnn_num_layers=1), lambda s, x: s.m(x, s.ei, s.batch))
    M('polynormer', lambda: G.Polynormer(F, 8, O, local_layers=1, global_layers=1, heads=2), lambda s, x: s.m(x, s.ei, s.batch))
    M('ar_link_predictor', lambda: G.ARLinkPredictor(F, 8, num_layers=1), lambda s, x: s.m(x, s.ei[:, :4]))
    M('gpse', lambda: G.GPSE(dim_in=F, dim_out=8, dim_inner=8, layers_pre_mp=1, layers_mp=1, layers_post_mp=1, num_node_targets=8, num_graph_targets=2, has_bn=False, dropout=0.0, virtual_node=False, multi_head_dim_inner=8), lambda s, x: s.m(DataC(x=x, edge_index=s.ei, batch=s.batch, y=torch.zeros(N, 8), y_graph=torch.zeros(2, 2)))[0])
    M('gpse_node_encoder', lambda: G.GPSENodeEncoder(dim_emb=8, dim_pe_in=F, dim_pe_out=4, dim_in=F, expand_x=True), lambda s, x: s.m(x, x))
    def tgn():
        m = G.TGNMemory(N, 3, 8, 8, message_module=G.models.tgn.IdentityMessage(3, 8, 8), aggregator_module=G.models.tgn.LastAggregator())
        m.eval()   # eval flushes the message store into the memory; then the memory is a lookup table
        with torch.no_grad(): m.memory.normal_(); m.last_update.copy_(torch.arange(N))
        return m
    M('tgn_memory', tgn, lambda s, x: flat(*s.m(s.nodes)), x=torch.zeros(1), nodes=torch.tensor([3, 1, 4, 1, 5, 7]))
    # ---- KGE models: score of 4 (head, relation, tail) triples
    def KG(n, mk): rcase(n, lambda s, x: s.m(s.h, s.r, s.t), lambda s, x: s.m(s.h, s.r, s.t), torch.zeros(1), lambda: dict(m=mk()), h=torch.tensor([0, 1, 2, 3]), r=torch.tensor([0, 1, 0, 1]), t=torch.tensor([4, 5, 6, 7]))
    KG('transe', lambda: G.TransE(N, 2, hidden_channels=O))
    KG('complex', lambda: G.ComplEx(N, 2, hidden_channels=O))
    KG('distmult', lambda: G.DistMult(N, 2, hidden_channels=O))
    KG('rotate', lambda: G.RotatE(N, 2, hidden_channels=O))
    # ---- encodings
    case('positional_encoding', lambda s, x: s.m(x), torch.arange(N).float(), lambda: dict(m=G.PositionalEncoding(O)))
    case('temporal_encoding', lambda s, x: s.m(x), torch.arange(N).float(), lambda: dict(m=G.TemporalEncoding(O)))
    # ---- functional
    def bro(s, x):   # bro: mean over graphs of || X_g X_g^T - I ||_F (X_g the graph's nodes, here 4 each): the per-graph block reshaped
        xd = x.view(2, 4, F); g = xd @ xd.transpose(1, 2) - torch.eye(4)
        return (g ** 2).sum((1, 2)).sqrt().mean().reshape(1)
    rcase('bro', bro, lambda s, x: G.functional.bro(x, s.batch).reshape(1), None, batch=BATCH)
    case('gini', lambda s, x: G.functional.gini(x).reshape(1), None)
    # ---- dense convolutional / pooling layers on the (1, N, N) adjacency
    def DN(n, mk, f=lambda s, x: s.m(x.unsqueeze(0), s.adj).squeeze(0), **b): rcase(n, f, f, None, lambda: dict(m=mk()), adj=LB.adj(EI, N).unsqueeze(0), **b)
    DN('dense_gcn_conv', lambda: G.DenseGCNConv(F, O), lambda s, x: s.m(x.unsqueeze(0), s.adj_l, add_loop=False).squeeze(0), adj_l=(LB.adj(EI, N) * (1 - torch.eye(N)) + torch.eye(N)).unsqueeze(0))   # the loops set on the adjacency in place of adj[:, idx, idx] = 1
    DN('dense_gin_conv', lambda: G.DenseGINConv(nn.Sequential(nn.Linear(F, O), nn.ReLU(), nn.Linear(O, O))))
    DN('dense_graph_conv', lambda: G.DenseGraphConv(F, O))
    DN('dense_sage_conv', lambda: G.DenseSAGEConv(F, O))
    DN('dense_gat_conv', lambda: G.DenseGATConv(F, O, heads=2), lambda s, x: s.m(x.unsqueeze(0), s.adj_l, add_loop=False).squeeze(0), adj_l=(LB.adj(EI, N) * (1 - torch.eye(N)) + torch.eye(N)).unsqueeze(0))
    case('dense_diff_pool', lambda s, x: flat(*G.dense_diff_pool(x.unsqueeze(0), s.adj, s.S)), None, adj=LB.adj(EI, N).unsqueeze(0), S=R(1, N, 3))
    def mincut(s, x):   # dense_mincut_pool with out_adj[:, ind, ind] = 0 as a (1 - I) multiply
        xb = x.unsqueeze(0); adj = s.adj; S = s.S.softmax(-1); k = 3
        out = S.transpose(1, 2) @ xb; out_adj = S.transpose(1, 2) @ adj @ S
        mincut_num = torch.einsum('ijj->i', out_adj); d_flat = torch.einsum('ijk->ij', adj); d = torch.eye(N) * d_flat.unsqueeze(2)
        mincut_den = torch.einsum('ijj->i', S.transpose(1, 2) @ d @ S); mincut_loss = -(mincut_num / mincut_den).mean()
        ss = S.transpose(1, 2) @ S; i_s = torch.eye(k); ortho_loss = (ss / torch.norm(ss, dim=(-1, -2), keepdim=True) - i_s / torch.norm(i_s)).norm(dim=(-1, -2)).mean()
        out_adj = out_adj * (1 - torch.eye(k)); d = torch.einsum('ijk->ij', out_adj); d = torch.sqrt(d)[:, None] + 1e-15; out_adj = (out_adj / d) / d.transpose(1, 2)
        return flat(out, out_adj, mincut_loss.reshape(1), ortho_loss.reshape(1))
    rcase('dense_mincut_pool', mincut, lambda s, x: flat(*G.dense_mincut_pool(x.unsqueeze(0), s.adj, s.S)), None, adj=LB.adj(EI, N).unsqueeze(0), S=R(1, N, 3))
    import torch_geometric.nn.dense.dmon_pool as DMP
    def dmon_forward(self, x, adj, mask=None):   # DMoNPooling.forward with out_adj[:, ind, ind] = 0 as a (1 - I) multiply
        x = x.unsqueeze(0) if x.dim() == 2 else x; adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        s = torch.softmax(self.mlp(x), dim=-1); (B_, Nn, _), C = x.size(), s.size(-1)
        out = NF.selu(s.transpose(1, 2) @ x); out_adj = s.transpose(1, 2) @ adj @ s
        degrees = torch.einsum('ijk->ij', adj).unsqueeze(-1); m = torch.einsum('ijk->i', degrees) / 2
        normalizer = (s.transpose(1, 2) @ degrees) @ (degrees.transpose(1, 2) @ s) / 2 / m.view(-1, 1, 1)
        spectral_loss = (-DMP._rank3_trace(out_adj - normalizer) / 2 / m).mean()
        ss = s.transpose(1, 2) @ s; i_s = torch.eye(C)
        ortho_loss = torch.norm(ss / torch.norm(ss, dim=(-1, -2), keepdim=True) - i_s / torch.norm(i_s), dim=(-1, -2)).mean()
        cluster_loss = (torch.norm(torch.einsum('ijk->ik', s), dim=1) / Nn * torch.norm(i_s) - 1).mean()
        out_adj = out_adj * (1 - torch.eye(C)); d = torch.sqrt(torch.einsum('ijk->ij', out_adj))[:, None] + 1e-15; out_adj = (out_adj / d) / d.transpose(1, 2)
        return s, out, out_adj, spectral_loss, ortho_loss, cluster_loss
    DMP.DMoNPooling.forward = dmon_forward
    DN('dmon_pooling', lambda: G.DMoNPooling(F, k=3), lambda s, x: flat(*s.m(x.unsqueeze(0), s.adj)))
    if name == '--list': return sorted(C)
    if name not in C: raise SystemExit('unknown case ' + name)
    return C[name]()

if __name__ == '__main__':
    if sys.argv[1] == '--list': print('\n'.join(build('--list'))); sys.exit(0)
    name, mlir_out, bin_out = sys.argv[1:4]
    m, x = build(name); m = m.eval()
    with torch.no_grad(): y = m.reference(x) if hasattr(m, 'reference') else m(x)
    with torch.no_grad(): assert torch.allclose(m(x), y, atol=getattr(m, 'atol', 1e-4), rtol=1e-4), 'exported body != reference'
    print('%s: in %s -> out %s' % (name, list(x.shape), list(y.shape)))
    record_searches(m, x); record_maxes(m); torch.Tensor.max = _fake_max; torch.atan2 = _static_atan2; _exporting = True
    mod = export_linalg(m, x)
    open(mlir_out, 'w').write(str(mod))
    xf = np.ascontiguousarray(x.numpy()).astype(np.float32).ravel(); yf = np.ascontiguousarray(y.numpy()).astype(np.float32).ravel()
    with open(bin_out, 'wb') as f:
        f.write(struct.pack('i', xf.size)); f.write(xf.tobytes()); f.write(struct.pack('i', yf.size)); f.write(yf.tobytes())
