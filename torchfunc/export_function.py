#!/usr/bin/env python3
"""Export one torch.nn.functional function (docs.pytorch.org/docs/2.14/nn.functional.html) through
torch-mlir to linalg-on-tensors, plus a PyTorch reference for one call. Every case is a single-input
module net(x) -> one float tensor (weights, second operands and loss targets are constant buffers), so the
generic host mod_main.c drives them all.       usage: export_function.py <fn> <mlir_out> <check_out>"""
import sys, struct, math, numpy as np, torch, torch.nn.functional as F
from torch_mlir import fx

class Case(torch.nn.Module):
    """net(x) = body(self, x); every tensor in `bufs` becomes a constant buffer of the module"""
    def __init__(s, body, **bufs):
        super().__init__(); s.body = body
        for k, v in bufs.items(): s.register_buffer(k, v)
    def forward(s, x): return s.body(s, x)

def R(*shape): return torch.randn(*shape)
class Ref(Case):
    """exported body differs from the reference: the reference (what the check.bin holds) is computed
    with the genuine torch.nn.functional call, the exported graph with an equivalent op composition
    that torch-mlir can lower (torch.export / torch-mlir reject the original op)."""
    def __init__(s, body, ref, **bufs): super().__init__(body, **bufs); s.ref = ref
    def reference(s, x): return s.ref(s, x)
def build(fn):
    torch.manual_seed(0)
    x1, x2, x3 = R(1, 4, 8), R(1, 4, 8, 8), R(1, 4, 4, 4, 4)      # 1-D / 2-D / 3-D feature maps
    v = R(1, 16)                                                   # a row vector for the activations
    C = {}
    def case(name, body, x, **bufs): C[name] = lambda: (Case(body, **bufs), x)
    def rcase(name, body, ref, x, **bufs): C[name] = lambda: (Ref(body, ref, **bufs), x)
    # ---- convolution
    case('conv1d', lambda s, x: F.conv1d(x, s.w, s.b, padding=1), x1, w=R(8, 4, 3), b=R(8))
    case('conv2d', lambda s, x: F.conv2d(x, s.w, s.b, padding=1), x2, w=R(8, 4, 3, 3), b=R(8))
    case('conv3d', lambda s, x: F.conv3d(x, s.w, s.b, padding=1), x3, w=R(8, 4, 3, 3, 3), b=R(8))
    case('conv_transpose1d', lambda s, x: F.conv_transpose1d(x, s.w, s.b, stride=2), x1, w=R(4, 8, 2), b=R(8))
    case('conv_transpose2d', lambda s, x: F.conv_transpose2d(x, s.w, s.b, stride=2), x2, w=R(4, 8, 2, 2), b=R(8))
    case('conv_transpose3d', lambda s, x: F.conv_transpose3d(x, s.w, s.b, stride=2), x3, w=R(4, 8, 2, 2, 2), b=R(8))
    case('unfold', lambda s, x: F.unfold(x, kernel_size=2, stride=2), x2)
    rcase('fold', lambda s, x: x.reshape(1, 4, 2, 2, 4, 4).permute(0, 1, 4, 2, 5, 3).reshape(1, 4, 8, 8),   # lowers to tm_tensor.scatter otherwise
          lambda s, x: F.fold(x, output_size=(8, 8), kernel_size=2, stride=2), R(1, 16, 16))
    # ---- pooling
    case('avg_pool1d', lambda s, x: F.avg_pool1d(x, 2), x1)
    case('avg_pool2d', lambda s, x: F.avg_pool2d(x, 2), x2)
    case('avg_pool3d', lambda s, x: F.avg_pool3d(x, 2), x3)
    case('max_pool1d', lambda s, x: F.max_pool1d(x, 2), x1)
    case('max_pool2d', lambda s, x: F.max_pool2d(x, 2), x2)
    case('max_pool3d', lambda s, x: F.max_pool3d(x, 2), x3)
    def unpool(d):   # max-pool (with indices) then max-unpool of the same tensor; the unpool's scatter
        pool, unp = getattr(F, f'max_pool{d}d'), getattr(F, f'max_unpool{d}d')   # is tm_tensor, so the
        def ref(s, x): y, i = pool(x, 2, return_indices=True); return unp(y, i, 2)   # export keeps x where it
        def body(s, x): return x * (x == F.interpolate(pool(x, 2), scale_factor=2, mode='nearest'))   # is the window max
        return body, ref
    for d, xx in ((1, x1), (2, x2), (3, x3)): rcase(f'max_unpool{d}d', *unpool(d), xx)
    case('lp_pool1d', lambda s, x: F.lp_pool1d(x, 2, 2), x1)
    case('lp_pool2d', lambda s, x: F.lp_pool2d(x, 2, 2), x2)
    case('lp_pool3d', lambda s, x: F.lp_pool3d(x, 2, 2), x3)
    case('adaptive_max_pool1d', lambda s, x: F.adaptive_max_pool1d(x, 2), x1)
    case('adaptive_max_pool2d', lambda s, x: F.adaptive_max_pool2d(x, 2), x2)
    case('adaptive_max_pool3d', lambda s, x: F.adaptive_max_pool3d(x, 1), x3)
    case('adaptive_avg_pool1d', lambda s, x: F.adaptive_avg_pool1d(x, 2), x1)
    case('adaptive_avg_pool2d', lambda s, x: F.adaptive_avg_pool2d(x, 2), x2)
    case('adaptive_avg_pool3d', lambda s, x: F.adaptive_avg_pool3d(x, 2), x3)
    rs2, rs3 = torch.rand(1, 4, 2), torch.rand(1, 4, 3)
    case('fractional_max_pool2d', lambda s, x: F.fractional_max_pool2d(x, 2, output_size=(4, 4), _random_samples=s.rs), x2, rs=rs2)
    case('fractional_max_pool3d', lambda s, x: F.fractional_max_pool3d(x, 2, output_size=(2, 2, 2), _random_samples=s.rs), x3, rs=rs3)
    # ---- attention
    case('scaled_dot_product_attention', lambda s, x: F.scaled_dot_product_attention(x, s.k, s.v), R(1, 2, 8, 16), k=R(1, 2, 8, 16), v=R(1, 2, 8, 16))
    # ---- activations (in-place variants run on a copy: the harness input must stay intact)
    acts = {'threshold': lambda x: F.threshold(x, 0.5, 0.0), 'threshold_': lambda x: F.threshold_(x.clone(), 0.5, 0.0),
            'relu': F.relu, 'relu_': lambda x: F.relu_(x.clone()), 'hardtanh': F.hardtanh, 'hardtanh_': lambda x: F.hardtanh_(x.clone()),
            'hardswish': F.hardswish, 'relu6': F.relu6, 'elu': F.elu, 'elu_': lambda x: F.elu_(x.clone()), 'selu': F.selu, 'celu': F.celu,
            'leaky_relu': lambda x: F.leaky_relu(x, 0.1), 'leaky_relu_': lambda x: F.leaky_relu_(x.clone(), 0.1),
            'rrelu': lambda x: F.rrelu(x, training=False), 'rrelu_': lambda x: F.rrelu_(x.clone(), training=False),
            'glu': lambda x: F.glu(x, dim=-1), 'gelu': F.gelu, 'logsigmoid': F.logsigmoid, 'hardshrink': F.hardshrink,
            'tanhshrink': F.tanhshrink, 'softsign': F.softsign, 'softplus': F.softplus, 'softmin': lambda x: F.softmin(x, dim=-1),
            'softmax': lambda x: F.softmax(x, dim=-1), 'softshrink': F.softshrink, 'log_softmax': lambda x: F.log_softmax(x, dim=-1),
            'tanh': torch.tanh, 'sigmoid': torch.sigmoid, 'hardsigmoid': F.hardsigmoid, 'silu': F.silu, 'mish': F.mish}
    for n, f in acts.items(): case(n, (lambda f: lambda s, x: f(x))(f), v)
    case('prelu', lambda s, x: F.prelu(x, s.w), v, w=torch.tensor([0.25]))
    # gumbel_softmax draws Gumbel noise inside: hard=False with the noise fixed by seeding is not
    # reproducible across export and reference, so feed the noise as a buffer through the formula
    case('gumbel_softmax', lambda s, x: F.softmax((x + s.g) / 0.5, dim=-1), v, g=-torch.empty(1, 16).exponential_().log())
    case('batch_norm', lambda s, x: F.batch_norm(x, s.rm, s.rv, s.w, s.b, training=False), x2, rm=R(4) * 0.3, rv=torch.rand(4) + 0.5, w=torch.rand(4) + 0.5, b=R(4) * 0.2)
    case('group_norm', lambda s, x: F.group_norm(x, 2, s.w, s.b), x2, w=torch.rand(4) + 0.5, b=R(4) * 0.2)
    case('instance_norm', lambda s, x: F.instance_norm(x, weight=s.w, bias=s.b), x2, w=torch.rand(4) + 0.5, b=R(4) * 0.2)
    case('layer_norm', lambda s, x: F.layer_norm(x, (16,), s.w, s.b), R(1, 4, 16), w=torch.rand(16) + 0.5, b=R(16) * 0.2)
    case('local_response_norm', lambda s, x: F.local_response_norm(x, 3), x2)
    case('rms_norm', lambda s, x: F.rms_norm(x, (16,), s.w), R(1, 4, 16), w=torch.rand(16) + 0.5)
    case('normalize', lambda s, x: F.normalize(x, dim=-1), v)
    # ---- linear
    case('linear', lambda s, x: F.linear(x, s.w, s.b), v, w=R(8, 16), b=R(8))
    case('bilinear', lambda s, x: F.bilinear(x, s.x2, s.w, s.b), v, x2=R(1, 8), w=R(4, 16, 8), b=R(4))
    # ---- dropout (eval: identity)
    for n in ('dropout', 'alpha_dropout', 'feature_alpha_dropout', 'dropout1d', 'dropout2d', 'dropout3d'):
        case(n, (lambda f: lambda s, x: f(x, 0.5, training=False))(getattr(F, n)), {'dropout1d': x1, 'dropout3d': x3}.get(n, x2))
    # ---- sparse (the float harness input is cast to long indices; one_hot's long result back to float)
    idx = torch.tensor([[1., 3., 0., 5.]])
    case('embedding', lambda s, x: F.embedding(x.long(), s.w), idx, w=R(10, 16))
    rcase('embedding_bag', lambda s, x: F.embedding(x.long(), s.w).mean(1),   # aten.embedding_bag has no lowering
          lambda s, x: F.embedding_bag(x.long(), s.w, mode='mean'), idx, w=R(10, 16))
    case('one_hot', lambda s, x: F.one_hot(x.long(), 8).float(), idx)
    # ---- distance
    case('pairwise_distance', lambda s, x: F.pairwise_distance(x, s.y), R(4, 8), y=R(4, 8))
    case('cosine_similarity', lambda s, x: F.cosine_similarity(x, s.y, dim=1), R(4, 8), y=R(4, 8))
    ii, jj = torch.triu_indices(4, 4, 1)
    rcase('pdist', lambda s, x: (x.index_select(0, s.ii) - x.index_select(0, s.jj)).norm(dim=1), lambda s, x: F.pdist(x), R(4, 8), ii=ii, jj=jj)
    # ---- losses: mean-reduced scalar, returned as a 1-element tensor; targets are buffers
    lg, pr, cls = R(4, 8), torch.rand(4, 8), torch.tensor([1, 0, 7, 3])
    def loss(name, body, x, **b): case(name, (lambda body: lambda s, x: body(s, x).reshape(1))(body), x, **b)
    loss('binary_cross_entropy', lambda s, x: F.binary_cross_entropy(torch.sigmoid(x), s.t), lg, t=pr)
    loss('binary_cross_entropy_with_logits', lambda s, x: F.binary_cross_entropy_with_logits(x, s.t), lg, t=pr)
    loss('poisson_nll_loss', lambda s, x: F.poisson_nll_loss(x, s.t), lg, t=torch.poisson(torch.rand(4, 8) * 3))
    loss('cosine_embedding_loss', lambda s, x: F.cosine_embedding_loss(x, s.y, s.t), lg, y=R(4, 8), t=torch.tensor([1., -1., 1., -1.]))
    loss('cross_entropy', lambda s, x: F.cross_entropy(x, s.t), lg, t=cls)
    loss('ctc_loss', lambda s, x: F.ctc_loss(F.log_softmax(x, -1), s.t, s.il, s.tl), R(6, 1, 5), t=torch.tensor([[1, 2, 2]]), il=torch.tensor([6]), tl=torch.tensor([3]))
    vr = torch.rand(4, 8) + 0.1
    rcase('gaussian_nll_loss', lambda s, x: (0.5 * (torch.log(s.var.clamp(min=1e-6)) + (x - s.t) ** 2 / s.var.clamp(min=1e-6))).mean().reshape(1),
          lambda s, x: F.gaussian_nll_loss(x, s.t, s.var).reshape(1), lg, t=R(4, 8), var=vr)   # F's var checks are data-dependent
    loss('hinge_embedding_loss', lambda s, x: F.hinge_embedding_loss(x, s.t), lg, t=torch.tensor([[1., -1.] * 4] * 4))
    loss('kl_div', lambda s, x: F.kl_div(F.log_softmax(x, -1), s.t, reduction='batchmean'), lg, t=F.softmax(R(4, 8), -1))
    loss('l1_loss', lambda s, x: F.l1_loss(x, s.t), lg, t=R(4, 8))
    if hasattr(F, 'linear_cross_entropy'):
        loss('linear_cross_entropy', lambda s, x: F.linear_cross_entropy(x, s.w, s.t), R(4, 16), w=R(8, 16), t=cls)
    loss('mse_loss', lambda s, x: F.mse_loss(x, s.t), lg, t=R(4, 8))
    loss('margin_ranking_loss', lambda s, x: F.margin_ranking_loss(x, s.y, s.t), R(8), y=R(8), t=torch.tensor([1., -1.] * 4))
    # multilabel_margin_loss: torch-mlir's decomposition of aten.multilabel_margin_loss (a chain of
    # index / select / gather generics) compiles but computes a wrong value on Hwacha, while every piece
    # of it does the right thing on its own (see known-issues/); the export uses the closed form with
    # the target sets as float masks: mean_b sum_{i in T_b, j not in T_b} relu(1 - x_bi + x_bj) / C
    mlT = torch.tensor([[3, 0, -1, 0, 0, 0, 0, 0]] * 4)
    pos = torch.arange(8); first = torch.where(mlT == -1, pos, torch.tensor(8)).min(dim=1).values
    isT = (pos[None, :, None] == torch.where(pos < first[:, None], mlT, torch.tensor(-1))[:, None, :]).any(-1).float()
    rcase('multilabel_margin_loss',
          lambda s, x: ((torch.clamp(1 - x[:, :, None] + x[:, None, :], min=0) * (s.isT[:, :, None] * (1 - s.isT)[:, None, :])).sum(dim=(1, 2)) / 8).mean().reshape(1),
          lambda s, x: F.multilabel_margin_loss(x, s.t).reshape(1), lg, t=mlT, isT=isT)
    loss('multilabel_soft_margin_loss', lambda s, x: F.multilabel_soft_margin_loss(x, s.t), lg, t=(torch.rand(4, 8) > 0.5).float())
    loss('multi_margin_loss', lambda s, x: F.multi_margin_loss(x, s.t), lg, t=cls)
    loss('nll_loss', lambda s, x: F.nll_loss(F.log_softmax(x, -1), s.t), lg, t=cls)
    loss('huber_loss', lambda s, x: F.huber_loss(x, s.t), lg, t=R(4, 8))
    loss('smooth_l1_loss', lambda s, x: F.smooth_l1_loss(x, s.t), lg, t=R(4, 8))
    loss('soft_margin_loss', lambda s, x: F.soft_margin_loss(x, s.t), lg, t=torch.tensor([[1., -1.] * 4] * 4))
    loss('triplet_margin_loss', lambda s, x: F.triplet_margin_loss(x, s.p, s.n), lg, p=R(4, 8), n=R(4, 8))
    loss('triplet_margin_with_distance_loss', lambda s, x: F.triplet_margin_with_distance_loss(x, s.p, s.n, distance_function=lambda a, b: 1 - F.cosine_similarity(a, b)), lg, p=R(4, 8), n=R(4, 8))
    # ---- vision
    case('pixel_shuffle', lambda s, x: F.pixel_shuffle(x, 2), R(1, 16, 4, 4))
    case('pixel_unshuffle', lambda s, x: F.pixel_unshuffle(x, 2), x2)
    case('pad', lambda s, x: F.pad(x, (1, 1, 1, 1)), x2)
    case('interpolate', lambda s, x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False), R(1, 4, 4, 4))
    case('upsample', lambda s, x: F.upsample(x, scale_factor=2, mode='nearest'), R(1, 4, 4, 4))
    case('upsample_nearest', lambda s, x: F.upsample_nearest(x, scale_factor=2), R(1, 4, 4, 4))
    case('upsample_bilinear', lambda s, x: F.upsample_bilinear(x, scale_factor=2), R(1, 4, 4, 4))
    theta = torch.tensor([[[0.8, 0.2, 0.1], [-0.2, 0.9, 0.0]]])
    case('affine_grid', lambda s, x: F.affine_grid(x, (1, 4, 8, 8), align_corners=False), theta)
    case('grid_sample', lambda s, x: F.grid_sample(x, s.g, align_corners=False), x2, g=F.affine_grid(theta, (1, 4, 8, 8), align_corners=False))
    # ---- low precision (fp8 / grouped GEMM; CPU support is partial, recorded as attempted)
    if hasattr(F, 'grouped_mm'):
        case('grouped_mm', lambda s, x: F.grouped_mm(x, s.w, offs=s.o), R(8, 16), w=R(2, 16, 8), o=torch.tensor([4, 8], dtype=torch.int32))
    if hasattr(F, 'scaled_mm'):
        case('scaled_mm', lambda s, x: F.scaled_mm(x.to(torch.float8_e4m3fn), s.w.to(torch.float8_e4m3fn).t(), s.sa, s.sb, out_dtype=torch.float32), R(8, 16), w=R(8, 16), sa=torch.ones(8, 1), sb=torch.ones(1, 8))
    if hasattr(F, 'scaled_grouped_mm'):
        case('scaled_grouped_mm', lambda s, x: F.scaled_grouped_mm(x.to(torch.float8_e4m3fn), s.w.to(torch.float8_e4m3fn), s.sa, s.sb, offs=s.o, out_dtype=torch.float32), R(8, 16), w=R(2, 8, 16), sa=torch.ones(8), sb=torch.ones(2, 8), o=torch.tensor([4, 8], dtype=torch.int32))
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
