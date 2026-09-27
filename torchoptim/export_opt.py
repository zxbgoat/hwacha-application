#!/usr/bin/env python3
"""Export one torch.optim interface (docs.pytorch.org/docs/2.14/optim.html) through torch-mlir to
linalg-on-tensors, plus a PyTorch reference. Every case runs K optimizer steps on a small quadratic
problem: parameters p (a 4x4 matrix and a 16-vector, the input), loss = 1/2 sum w (p - t)^2 with
constant w > 0 and targets t, so the gradient is w (p - t) in closed form and the exported graph holds
the update rule only (torch.export takes no autograd). The exported body calls the algorithm's
functional single-tensor implementation (torch.optim.<algo>.<algo>, what Optimizer.step runs); the
reference drives the real Optimizer / LRScheduler / AveragedModel objects with autograd on the same
problem, and the parameters after K steps are compared.  usage: export_opt.py <name> <mlir> <check>"""
import sys, os, math, struct, numpy as np, torch, torch.nn as nn, torch.nn.functional as NF
import torch.optim as O, torch.optim.lr_scheduler as LS, torch.optim.swa_utils as SWA
import importlib
# the functional single-tensor implementations (torch.optim.<algo>.<algo>, what Optimizer.step runs)
FN = {n.lstrip('_'): getattr(importlib.import_module('torch.optim.' + n), n.lstrip('_')) for n in ('adadelta', '_adafactor', 'adagrad', 'adam', 'adamw', 'adamax', 'asgd', '_muon', 'nadam', 'radam', 'rmsprop', 'rprop', 'sgd')}
from torch_mlir import fx

class Case(nn.Module):
    def __init__(s, body, mods=None, **bufs):
        super().__init__(); s.body = body
        for k, v in (mods() if callable(mods) else (mods or {})).items(): setattr(s, k, v)
        for k, v in bufs.items(): s.register_buffer(k, v)
    def forward(s, x): return s.body(s, x)
class Ref(Case):
    def __init__(s, body, ref, mods=None, _atol=1e-4, **bufs): super().__init__(body, mods, **bufs); s.ref = ref; s.atol = _atol
    def reference(s, x): return s.ref(s, x)
def export_linalg(m, x):
    from torch_mlir.extras.fx_decomp_util import get_decomposition_table
    table = {k: v for k, v in get_decomposition_table().items() if not any(str(k).startswith('aten.' + n) for n in ('zeros', 'ones', 'full', 'new_zeros', 'new_ones', 'new_full', 'empty_like', 'zeros_like', 'ones_like', 'full_like'))}
    return fx.export_and_import(torch.export.export(m, (x,)), x, output_type='linalg-on-tensors', func_name='net', decomposition_table=table)

# ---- the problem: two parameters (4x4, 16), loss = 1/2 sum w (p - t)^2, K steps
K = 4
SHAPES = [(4, 4), (16,)]
def split(x):
    """the input (32,) -> the two parameters, as fresh tensors (the functional updates are in place: on views of
    the input they would export as input mutations)"""
    return [x[:16].reshape(4, 4).clone(), x[16:].clone()]
def grads(s, ps):
    return [s.w0 * (ps[0] - s.t0), s.w1 * (ps[1] - s.t1)]
def loss_of(s, ps):
    return 0.5 * ((s.w0 * (ps[0] - s.t0) ** 2).sum() + (s.w1 * (ps[1] - s.t1) ** 2).sum())
def problem():
    torch.manual_seed(0)
    return dict(w0=torch.rand(4, 4) + 0.5, t0=torch.randn(4, 4), w1=torch.rand(16) + 0.5, t1=torch.randn(16))

def run_optimizer(s, x, make_opt, sched=None, steps=K, closure=False):
    """the reference: the real Optimizer (and scheduler) on nn.Parameters with autograd"""
    ps = [nn.Parameter(t.clone()) for t in split(x)]
    opt = make_opt(ps); sc = sched(opt) if sched else None
    for _ in range(steps):
        if closure:
            def cl():
                opt.zero_grad(); l = loss_of(s, ps); l.backward(); return l
            opt.step(cl)
        else:
            opt.zero_grad(); loss_of(s, ps).backward(); opt.step()
        if sc is not None: sc.step()
    return torch.cat([p.detach().reshape(-1) for p in ps])
def flat(ps): return torch.cat([p.reshape(-1) for p in ps])
def zeros_like_ps(): return [torch.zeros(*sh) for sh in SHAPES]

def build(name):
    torch.manual_seed(0)
    C = {}
    def rcase(n, body, ref, mods=None, **b): C[n] = lambda: (Ref(body, ref, mods, **problem(), **b), torch.randn(32))
    def algo(n, body, make_opt, **kw): rcase(n, body, lambda s, x: run_optimizer(s, x, make_opt, **kw))
    # ---- algorithms: the functional single-tensor path with the same hyper-parameters as the Optimizer
    def sgd(momentum=0.0, dampening=0.0, nesterov=False, weight_decay=0.0, lr=0.1, maximize=False):
        def body(s, x):
            ps = split(x); bufs = [None, None]
            for _ in range(K):
                FN['sgd'](ps, grads(s, ps), bufs, has_sparse_grad=False, foreach=False, fused=False, grad_scale=None, found_inf=None, weight_decay=weight_decay, momentum=momentum, lr=lr, dampening=dampening, nesterov=nesterov, maximize=maximize)
            return flat(ps)
        return body
    algo('sgd', sgd(), lambda ps: O.SGD(ps, lr=0.1))
    algo('sgd_momentum', sgd(momentum=0.9, weight_decay=0.01), lambda ps: O.SGD(ps, lr=0.1, momentum=0.9, weight_decay=0.01))
    algo('sgd_nesterov', sgd(momentum=0.9, nesterov=True), lambda ps: O.SGD(ps, lr=0.1, momentum=0.9, nesterov=True))
    algo('sgd_maximize', sgd(maximize=True, lr=0.01), lambda ps: O.SGD(ps, lr=0.01, maximize=True))
    def adam_like(fn, amsgrad=False, weight_decay=0.0, lr=0.1, decoupled=False, maximize=False, **extra):
        def body(s, x):
            ps = split(x); m = zeros_like_ps(); v = zeros_like_ps(); mx = zeros_like_ps() if amsgrad else []; st = [torch.tensor(0.0), torch.tensor(0.0)]
            for _ in range(K):
                fn(ps, grads(s, ps), m, v, mx, st, foreach=False, capturable=False, differentiable=False, fused=False, grad_scale=None, found_inf=None, has_complex=False, amsgrad=amsgrad, beta1=0.9, beta2=0.999, lr=lr, weight_decay=weight_decay, eps=1e-8, maximize=maximize, **extra)
            return flat(ps)
        return body
    algo('adam', adam_like(FN['adam'], decoupled_weight_decay=False), lambda ps: O.Adam(ps, lr=0.1))
    algo('adam_amsgrad', adam_like(FN['adam'], amsgrad=True, weight_decay=0.01, decoupled_weight_decay=False), lambda ps: O.Adam(ps, lr=0.1, amsgrad=True, weight_decay=0.01))
    algo('adamw', adam_like(FN['adamw'], weight_decay=0.01), lambda ps: O.AdamW(ps, lr=0.1, weight_decay=0.01))
    algo('adamw_amsgrad', adam_like(FN['adamw'], amsgrad=True, weight_decay=0.01), lambda ps: O.AdamW(ps, lr=0.1, weight_decay=0.01, amsgrad=True))
    def adamax_body(s, x):
        ps = split(x); m = zeros_like_ps(); u = zeros_like_ps(); st = [torch.tensor(0.0), torch.tensor(0.0)]
        for _ in range(K): FN['adamax'](ps, grads(s, ps), m, u, st, foreach=False, maximize=False, differentiable=False, capturable=False, has_complex=False, eps=1e-8, beta1=0.9, beta2=0.999, lr=0.1, weight_decay=0.0)
        return flat(ps)
    algo('adamax', adamax_body, lambda ps: O.Adamax(ps, lr=0.1))
    def adadelta_body(s, x):
        ps = split(x); sq = zeros_like_ps(); ad = zeros_like_ps(); st = [torch.tensor(0.0), torch.tensor(0.0)]
        for _ in range(K): FN['adadelta'](ps, grads(s, ps), sq, ad, st, capturable=False, foreach=False, differentiable=False, has_complex=False, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0.0, maximize=False)
        return flat(ps)
    algo('adadelta', adadelta_body, lambda ps: O.Adadelta(ps, lr=1.0))
    def adagrad_body(s, x):
        ps = split(x); ss = zeros_like_ps(); st = [torch.tensor(0.0), torch.tensor(0.0)]
        for _ in range(K): FN['adagrad'](ps, grads(s, ps), ss, st, fused=False, grad_scale=None, found_inf=None, has_sparse_grad=False, foreach=False, differentiable=False, has_complex=False, lr=0.1, weight_decay=0.0, lr_decay=0.0, eps=1e-10, maximize=False)
        return flat(ps)
    algo('adagrad', adagrad_body, lambda ps: O.Adagrad(ps, lr=0.1))
    def nadam_body(s, x):
        ps = split(x); m = zeros_like_ps(); v = zeros_like_ps(); mu = [torch.tensor(1.0), torch.tensor(1.0)]; st = [torch.tensor(0.0), torch.tensor(0.0)]
        for _ in range(K): FN['nadam'](ps, grads(s, ps), m, v, mu, st, decoupled_weight_decay=False, foreach=False, capturable=False, differentiable=False, has_complex=False, maximize=False, beta1=0.9, beta2=0.999, lr=0.1, weight_decay=0.0, momentum_decay=4e-3, eps=1e-8)
        return flat(ps)
    algo('nadam', nadam_body, lambda ps: O.NAdam(ps, lr=0.1))
    def radam_body(s, x):
        ps = split(x); m = zeros_like_ps(); v = zeros_like_ps(); st = [torch.tensor(0.0), torch.tensor(0.0)]
        for _ in range(K): FN['radam'](ps, grads(s, ps), m, v, st, decoupled_weight_decay=False, foreach=False, differentiable=False, capturable=False, has_complex=False, maximize=False, beta1=0.9, beta2=0.999, lr=0.1, weight_decay=0.0, eps=1e-8)
        return flat(ps)
    algo('radam', radam_body, lambda ps: O.RAdam(ps, lr=0.1))
    def rmsprop_body(centered=False, momentum=0.0):
        def body(s, x):
            ps = split(x); sq = zeros_like_ps(); ga = zeros_like_ps() if centered else []; mb = zeros_like_ps() if momentum else []; st = [torch.tensor(0.0), torch.tensor(0.0)]
            for _ in range(K): FN['rmsprop'](ps, grads(s, ps), sq, ga, mb, st, foreach=False, maximize=False, differentiable=False, capturable=False, has_complex=False, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=momentum, centered=centered)
            return flat(ps)
        return body
    algo('rmsprop', rmsprop_body(), lambda ps: O.RMSprop(ps, lr=0.01))
    algo('rmsprop_centered', rmsprop_body(centered=True, momentum=0.9), lambda ps: O.RMSprop(ps, lr=0.01, centered=True, momentum=0.9))
    def rprop_body(s, x):   # _single_tensor_rprop with its sign bookkeeping (masked assignments, which lower to tm_tensor.scan) as torch.where
        ps = split(x); pv = zeros_like_ps(); ss = [torch.full(sh, 0.01) for sh in SHAPES]; smin, smax, em, ep = 1e-6, 50, 0.5, 1.2
        for _ in range(K):
            gs = grads(s, ps)
            for i in range(2):
                sign = (gs[i] * pv[i]).sign(); sign = torch.where(sign > 0, torch.full_like(sign, ep), sign); sign = torch.where(sign < 0, torch.full_like(sign, em), sign); sign = torch.where(sign == 0, torch.ones_like(sign), sign)
                ss[i] = (ss[i] * sign).clamp(smin, smax); g = torch.where(sign == em, torch.zeros_like(gs[i]), gs[i])
                ps[i] = ps[i] - g.sign() * ss[i]; pv[i] = g
        return flat(ps)
    algo('rprop', rprop_body, lambda ps: O.Rprop(ps, lr=0.01))
    def asgd_body(s, x):
        ps = split(x); ax = zeros_like_ps(); mus = [torch.tensor(1.0), torch.tensor(1.0)]; etas = [torch.tensor(0.1), torch.tensor(0.1)]; st = [torch.tensor(0.0), torch.tensor(0.0)]
        for _ in range(K): FN['asgd'](ps, grads(s, ps), ax, mus, etas, st, foreach=False, maximize=False, differentiable=False, capturable=False, has_complex=False, lambd=1e-4, lr=0.1, t0=1e6, alpha=0.75, weight_decay=0.0)
        return flat(ps)
    algo('asgd', asgd_body, lambda ps: O.ASGD(ps, lr=0.1))
    def adafactor_body(s, x):   # _single_tensor_adafactor with its .item() scalars (the parameter's RMS for alpha, the update's RMS for the clipping) kept as tensors
        ps = split(x); rv = torch.zeros(4, 1); cv = torch.zeros(1, 4); va = torch.zeros(16); lr, b2d, eps1, eps2, d = 0.01, -0.8, torch.finfo(torch.float32).eps, 1e-3, 1.0
        for step in range(1, K + 1):
            omb = step ** b2d; rho = min(lr, 1 / step ** 0.5); gs = grads(s, ps)
            for i in range(2):
                p, g = ps[i], gs[i]
                alpha = torch.clamp(p.norm(2) / p.numel() ** 0.5, min=eps2) * rho
                if g.dim() > 1:
                    rv = rv + omb * (g.norm(dim=-1, keepdim=True).square() / g.shape[-1] - rv); cv = cv + omb * (g.norm(dim=-2, keepdim=True).square() / g.shape[-2] - cv)
                    ve = (rv @ cv) / rv.mean(dim=-2, keepdim=True).clamp(min=eps1)
                else:
                    va = va + omb * (g * g - va); ve = va
                upd = ve.clamp(min=eps1 * eps1).rsqrt() * g
                den = torch.clamp(upd.norm(2) / (upd.numel() ** 0.5 * d), min=1.0)
                ps[i] = p - upd * (alpha / den)
        return flat(ps)
    algo('adafactor', adafactor_body, lambda ps: O.Adafactor(ps, lr=0.01))
    # Muon's Newton-Schulz orthogonalisation runs in bfloat16 (grad.bfloat16()), which the RISC-V toolchain has no
    # support for (__truncsfbf2); both the export and the reference run it in float32 (the same algorithm, the cast dropped)
    import torch.optim._muon as MU
    _ns = MU._zeropower_via_newtonschulz
    def ns32(grad, ns_coefficients, ns_steps, eps):
        a, b, c = ns_coefficients; og = grad.clone()
        if grad.size(0) > grad.size(1): og = og.T
        og = og / og.norm().clamp(min=eps)
        for _ in range(ns_steps):
            gm = og @ og.T; gu = torch.addmm(gm, gm, gm, beta=b, alpha=c); og = torch.addmm(og, gu, og, beta=a)
        return og.T if grad.size(0) > grad.size(1) else og
    MU._zeropower_via_newtonschulz = ns32
    def muon_body(s, x):   # Muon takes 2-D parameters: the 4x4 matrix (Newton-Schulz orthogonalisation) and the vector as a 1x16 matrix
        ps = [x[:16].reshape(4, 4).clone(), x[16:].reshape(1, 16).clone()]; gs = lambda ps: [s.w0 * (ps[0] - s.t0), (s.w1 * (ps[1].reshape(16) - s.t1)).reshape(1, 16)]; mb = [torch.zeros(4, 4), torch.zeros(1, 16)]
        for _ in range(K): FN['muon'](ps, gs(ps), mb, foreach=False, lr=0.02, weight_decay=0.0, momentum=0.95, nesterov=True, ns_coefficients=(3.4445, -4.775, 2.0315), ns_steps=5, eps=1e-7, adjust_lr_fn=None, has_complex=False)
        return flat(ps)
    def muon_ref(s, x):
        ps = [nn.Parameter(x[:16].reshape(4, 4).clone()), nn.Parameter(x[16:].reshape(1, 16).clone())]; opt = O.Muon(ps, lr=0.02, weight_decay=0.0)
        for _ in range(K):
            opt.zero_grad(); (0.5 * ((s.w0 * (ps[0] - s.t0) ** 2).sum() + (s.w1 * (ps[1].reshape(16) - s.t1) ** 2).sum())).backward(); opt.step()
        return flat([p.detach() for p in ps])
    rcase('muon', muon_body, muon_ref)
    # SparseAdam: the Adam update on the non-zero rows of a sparse gradient (the moments of the rows touched); here on
    # the dense problem every row is touched, so it is Adam's update with the row mask 1 -- exported through torch.optim._functional.sparse_adam's math written densely
    def sparse_adam_body(s, x):
        ps = split(x); m = zeros_like_ps(); v = zeros_like_ps(); b1, b2, lr, eps = 0.9, 0.999, 0.1, 1e-8
        for step in range(1, K + 1):
            for i, g in enumerate(grads(s, ps)):
                m[i] = m[i] + (1 - b1) * (g - m[i]); v[i] = v[i] + (1 - b2) * (g * g - v[i])
                bc1 = 1 - b1 ** step; bc2 = 1 - b2 ** step; ps[i] = ps[i] - (lr / bc1) * (m[i] / bc1 * bc1) / (v[i] / bc2).sqrt().add(eps) if False else ps[i] - lr * math.sqrt(bc2) / bc1 * m[i] / (v[i].sqrt() + eps)
        return flat(ps)
    def sparse_adam_ref(s, x):
        ps = [nn.Parameter(t.clone()) for t in split(x)]; opt = O.SparseAdam(ps, lr=0.1)
        for _ in range(K):
            opt.zero_grad(); loss_of(s, ps).backward()
            for p in ps: p.grad = p.grad.to_sparse()
            opt.step()
        return flat([p.detach() for p in ps])
    rcase('sparse_adam', sparse_adam_body, sparse_adam_ref)
    # LBFGS: history 4, 3 iterations per step with the closure, no line search: unrolled two-loop recursion
    def lbfgs_body(s, x):   # LBFGS.step (lr 0.1, max_iter 3 -> max_eval 3, history 4, no line search, tolerances 0) unrolled over K closure calls,
        ps = split(x); lr, max_iter, hist = 0.1, 3, 4          # the global state (direction, step, history, prev gradient, n_iter) carried across the steps
        g = lambda: flat(grads(s, ps))
        old_dirs, old_stps, ro = [], [], []; H_diag = None; d = None; t = None; prev_g = None; n_total = 0
        for _ in range(K):
            fg = g(); evals = 1
            for it in range(max_iter):
                n_total += 1
                if n_total == 1: d = fg.neg(); H_diag = torch.tensor(1.0)
                else:
                    y = fg - prev_g; sv = d * t; ys = (y * sv).sum()   # ys > 1e-10 holds on this problem (checked against the reference)
                    if len(old_dirs) == hist: old_dirs.pop(0); old_stps.pop(0); ro.pop(0)
                    old_dirs.append(y); old_stps.append(sv); ro.append(1.0 / ys); H_diag = ys / (y * y).sum()
                    q = fg.neg(); al = [None] * len(old_dirs)
                    for i in range(len(old_dirs) - 1, -1, -1): al[i] = (old_stps[i] * q).sum() * ro[i]; q = q - al[i] * old_dirs[i]
                    d = q * H_diag
                    for i in range(len(old_dirs)): d = d + old_stps[i] * (al[i] - (old_dirs[i] * d).sum() * ro[i])
                prev_g = fg.clone()
                t = torch.clamp(1.0 / fg.abs().sum(), max=1.0) * lr if n_total == 1 else lr
                ps = split(flat(ps) + t * d)
                if it != max_iter - 1: fg = g(); evals += 1
                if evals >= 3: break   # max_eval = max_iter * 5 / 4 = 3
        return flat(ps)
    algo('lbfgs', lbfgs_body, lambda ps: O.LBFGS(ps, lr=0.1, max_iter=3, history_size=4, tolerance_grad=0, tolerance_change=0), closure=True)
    # ---- learning-rate schedulers: SGD (lr 0.1) for 8 steps, the scheduler stepped after each; the exported graph
    # takes the schedule's learning rates (Python floats the scheduler computes, constants of the case) step by step
    KS = 8
    def sched_lrs(make_sched, base_lr=0.1, opt_fn=None, steps=KS):
        p = nn.Parameter(torch.zeros(1)); opt = (opt_fn or (lambda ps: O.SGD(ps, lr=base_lr)))([p]); sc = make_sched(opt); lrs = []
        for _ in range(steps): lrs.append(opt.param_groups[0]['lr']); opt.step(); sc.step()
        return lrs
    def sched(n, make_sched, base_lr=0.1, opt_fn=None):
        lrs = sched_lrs(make_sched, base_lr, opt_fn)
        def body(s, x):
            ps = split(x)
            for lr in lrs: FN['sgd'](ps, grads(s, ps), [None, None], has_sparse_grad=False, foreach=False, fused=False, grad_scale=None, found_inf=None, weight_decay=0.0, momentum=0.0, lr=lr, dampening=0.0, nesterov=False, maximize=False)
            return flat(ps)
        rcase(n, body, lambda s, x: run_optimizer(s, x, opt_fn or (lambda ps: O.SGD(ps, lr=base_lr)), make_sched, steps=KS), lrs=torch.tensor(lrs))
    sched('lambda_lr', lambda o: LS.LambdaLR(o, lambda e: 0.9 ** e))
    sched('multiplicative_lr', lambda o: LS.MultiplicativeLR(o, lambda e: 0.9))
    sched('step_lr', lambda o: LS.StepLR(o, step_size=3, gamma=0.5))
    sched('multi_step_lr', lambda o: LS.MultiStepLR(o, milestones=[2, 5], gamma=0.5))
    sched('constant_lr', lambda o: LS.ConstantLR(o, factor=0.5, total_iters=4))
    sched('linear_lr', lambda o: LS.LinearLR(o, start_factor=0.25, total_iters=4))
    sched('exponential_lr', lambda o: LS.ExponentialLR(o, gamma=0.8))
    sched('polynomial_lr', lambda o: LS.PolynomialLR(o, total_iters=6, power=2.0))
    sched('cosine_annealing_lr', lambda o: LS.CosineAnnealingLR(o, T_max=4, eta_min=0.01))
    sched('chained_scheduler', lambda o: LS.ChainedScheduler([LS.ConstantLR(o, factor=0.5, total_iters=3), LS.ExponentialLR(o, gamma=0.9)]))
    sched('sequential_lr', lambda o: LS.SequentialLR(o, [LS.ConstantLR(o, factor=0.5, total_iters=3), LS.ExponentialLR(o, gamma=0.9)], milestones=[3]))
    sched('cyclic_lr', lambda o: LS.CyclicLR(o, base_lr=0.01, max_lr=0.1, step_size_up=3, cycle_momentum=False))
    sched('one_cycle_lr', lambda o: LS.OneCycleLR(o, max_lr=0.1, total_steps=KS + 1, cycle_momentum=False))
    sched('cosine_annealing_warm_restarts', lambda o: LS.CosineAnnealingWarmRestarts(o, T_0=3, T_mult=2, eta_min=0.01))
    # ReduceLROnPlateau steps on the loss: the reference feeds the true loss; the schedule of the case's own run is a constant
    def plateau_lrs():
        s0 = Ref(lambda s, x: x, lambda s, x: x, None, **problem()); x = torch.randn(32); torch.manual_seed(0)
        ps = [nn.Parameter(t.clone()) for t in split(x)]; opt = O.SGD(ps, lr=0.1); sc = LS.ReduceLROnPlateau(opt, factor=0.5, patience=0, threshold=0.5, threshold_mode='rel'); lrs = []
        for _ in range(KS):
            lrs.append(opt.param_groups[0]['lr']); opt.zero_grad(); l = loss_of(s0, ps); l.backward(); opt.step(); sc.step(l.item())
        return lrs
    def plateau_ref(s, x):
        ps = [nn.Parameter(t.clone()) for t in split(x)]; opt = O.SGD(ps, lr=0.1); sc = LS.ReduceLROnPlateau(opt, factor=0.5, patience=0, threshold=0.5, threshold_mode='rel')
        for _ in range(KS):
            opt.zero_grad(); l = loss_of(s, ps); l.backward(); opt.step(); sc.step(l.item())
        return flat([p.detach() for p in ps])
    plrs = plateau_lrs()
    def plateau_body(s, x):
        ps = split(x)
        for lr in plrs: FN['sgd'](ps, grads(s, ps), [None, None], has_sparse_grad=False, foreach=False, fused=False, grad_scale=None, found_inf=None, weight_decay=0.0, momentum=0.0, lr=lr, dampening=0.0, nesterov=False, maximize=False)
        return flat(ps)
    rcase('reduce_lr_on_plateau', plateau_body, plateau_ref, lrs=torch.tensor(plrs))
    # ---- weight averaging: SGD steps with the averaged copy updated after each (AveragedModel with the SWA / EMA
    # average functions), SWALR annealing the learning rate to swa_lr; the output is [parameters | averaged parameters]
    def averaged(n, avg_kind, swalr=False):
        lrs = sched_lrs(lambda o: SWA.SWALR(o, swa_lr=0.02, anneal_epochs=3, anneal_strategy='linear'), 0.1) if swalr else [0.1] * KS
        def body(s, x):
            ps = split(x); avg = None; n_avg = 0
            for lr in lrs:
                FN['sgd'](ps, grads(s, ps), [None, None], has_sparse_grad=False, foreach=False, fused=False, grad_scale=None, found_inf=None, weight_decay=0.0, momentum=0.0, lr=lr, dampening=0.0, nesterov=False, maximize=False)
                if n_avg == 0: avg = [p.clone() for p in ps]                       # AveragedModel's first update copies the parameters
                elif avg_kind == 'swa': avg = [a + (p - a) / (n_avg + 1) for a, p in zip(avg, ps)]
                else: avg = [0.9 * a + 0.1 * p for a, p in zip(avg, ps)]
                n_avg += 1
            return torch.cat([flat(ps), flat(avg)])
        def ref(s, x):
            mod = nn.Module(); mod.a = nn.Parameter(x[:16].reshape(4, 4).clone()); mod.b = nn.Parameter(x[16:].clone()); ps = [mod.a, mod.b]
            am = SWA.AveragedModel(mod, multi_avg_fn=SWA.get_swa_multi_avg_fn() if avg_kind == 'swa' else SWA.get_ema_multi_avg_fn(0.9)) if 'multi' in n else SWA.AveragedModel(mod, avg_fn=SWA.get_swa_avg_fn() if avg_kind == 'swa' else SWA.get_ema_avg_fn(0.9))
            opt = O.SGD(ps, lr=0.1); sc = SWA.SWALR(opt, swa_lr=0.02, anneal_epochs=3, anneal_strategy='linear') if swalr else None
            for _ in range(KS):
                opt.zero_grad(); loss_of(s, ps).backward(); opt.step(); am.update_parameters(mod)
                if sc: sc.step()
            return torch.cat([flat([p.detach() for p in ps]), flat([am.module.a.detach(), am.module.b.detach()])])
        rcase(n, body, ref, lrs=torch.tensor(lrs))
    averaged('averaged_model_swa', 'swa'); averaged('averaged_model_ema', 'ema')
    averaged('swa_multi_avg_fn', 'swa'); averaged('ema_multi_avg_fn', 'ema')
    averaged('swalr', 'swa', swalr=True)
    if name == '--list': return sorted(C)
    if name not in C: raise SystemExit('unknown case ' + name)
    return C[name]()

if __name__ == '__main__':
    if sys.argv[1] == '--list': print('\n'.join(build('--list'))); sys.exit(0)
    name, mlir_out, bin_out = sys.argv[1:4]
    m, x = build(name); m = m.eval()
    y = m.reference(x).detach()
    with torch.no_grad(): assert torch.allclose(m(x), y, atol=getattr(m, 'atol', 1e-4), rtol=1e-4), 'exported body != reference: %g' % (m(x) - y).abs().max()
    print('%s: in %s -> out %s' % (name, list(x.shape), list(y.shape)))
    mod = export_linalg(m, x)
    open(mlir_out, 'w').write(str(mod))
    xf = np.ascontiguousarray(x.numpy()).astype(np.float32).ravel(); yf = np.ascontiguousarray(y.numpy()).astype(np.float32).ravel()
    with open(bin_out, 'wb') as f:
        f.write(struct.pack('i', xf.size)); f.write(xf.tobytes()); f.write(struct.pack('i', yf.size)); f.write(yf.tobytes())
