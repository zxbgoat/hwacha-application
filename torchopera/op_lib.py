"""Tensor re-implementations for the torch.* cases whose op has no torch-mlir / hwacha-mlir lowering
(tm_tensor.sort / scan / scatter, libm calls hwacha-cc has no vector version of, LAPACK ops, complex
tensors, data-dependent shapes). Every function is checked against the genuine op in export_op.py."""
import math, torch

# ---- elementary functions hwacha-cc has no vector kernel for (asinf / acosf / tanf / sinhf / coshf / asinhf /
# acoshf / atanhf / llvm.ceil / exp2 / log2 / llvm.pow): from the ones it has (atan, sqrt, exp, log, floor, sin, cos)
LN2 = math.log(2.0)
def asin(x): return torch.atan(x / torch.sqrt((1 - x * x).clamp(min=1e-30)))
def acos(x): return math.pi / 2 - asin(x)
def tan(x): return torch.sin(x) / torch.cos(x)
def sinh(x): return 0.5 * (torch.exp(x) - torch.exp(-x))
def cosh(x): return 0.5 * (torch.exp(x) + torch.exp(-x))
def asinh(x): return torch.log(x + torch.sqrt(x * x + 1))
def acosh(x): return torch.log(x + torch.sqrt(x * x - 1))
def atanh(x): return 0.5 * torch.log((1 + x) / (1 - x))
def ceil(x): return -torch.floor(-x)
def exp2(x): return torch.exp(x * LN2)
def log2(x): return torch.log(x) / LN2
def powf(x, y): return torch.exp(y * torch.log(x))


def atan2(y, x):
    ax = torch.where(x == 0, torch.full_like(x, 1e-30), x); a = torch.atan(y / ax)
    return torch.where(x < 0, a + torch.where(y >= 0, torch.full_like(a, math.pi), torch.full_like(a, -math.pi)), a)

def onehot(idx, n): return (idx[:, None] == torch.arange(n)[None, :]).float()
def pair_mask(pairs, M, N): return torch.zeros(M, N).index_put((pairs[0], pairs[1]), torch.ones(pairs.shape[1]))

# ---- ranks instead of sorts (random data: ties broken by index, as torch.sort's stable order)
def rank_rows(x):
    """rank of every element within its row (0 = the smallest): the inverse of argsort"""
    ar = torch.arange(x.shape[1])
    lt = (x[:, None, :] < x[:, :, None]) | ((x[:, None, :] == x[:, :, None]) & (ar[None, None, :] < ar[None, :, None]))   # [b, j, i]: i before j
    return lt.float().sum(-1)
def sort_rows(x, descending=False):
    r = rank_rows(-x if descending else x)                                        # (B, N)
    sel = (r[:, None, :] == torch.arange(x.shape[1]).float()[None, :, None]).float()   # (B, N, N): row k picks rank k
    return (sel * x[:, None, :]).sum(-1)
def inv_perm(p):
    out = torch.empty_like(p); out.scatter_(1, p, torch.arange(p.shape[1]).expand_as(p)); return out
def quantile_rank(x, q, interpolation='linear'):
    """torch.quantile(x, q, dim=1) through the ranks (also median / kthvalue with the right q and 'lower')"""
    n = x.shape[1]; pos = q * (n - 1); lo = math.floor(pos); hi = -math.floor(-pos); frac = pos - lo
    r = rank_rows(x)
    at = lambda k: ((r == k).float() * x).sum(1)
    if interpolation == 'lower': return at(lo)
    if interpolation == 'higher': return at(hi)
    return at(lo) + (at(hi) - at(lo)) * frac
def mode_rows(x):
    """the most frequent value of every row (the smallest one on ties, as torch.mode)"""
    cnt = (x[:, :, None] == x[:, None, :]).float().sum(-1)                        # (B, N): count of each element's value
    best = cnt.max(1, keepdim=True).values
    cand = torch.where(cnt == best, x, torch.full_like(x, float('inf')))
    return cand.min(1).values
def cummax_rows(x):
    """running maximum along the row: max over the triangular mask"""
    n = x.shape[1]; m = torch.tril(torch.ones(n, n)).bool()                       # m[k, j] = j <= k
    return torch.where(m[None], x[:, None, :], torch.full_like(x[:, None, :].expand(-1, n, -1), -1e30)).amax(-1)
def cumprod_rows(x):
    """cumulative product of positive rows: exp(cumsum(log))"""
    n = x.shape[1]; return torch.exp(torch.log(x) @ torch.triu(torch.ones(n, n)))
def unique_sorted(x):
    """torch.unique of a 1-D tensor whose distinct values are known (the case's constant): the sorted distinct values via ranks"""
    r = rank_rows(x[None])[0]                                                        # rank with index tie-break
    first = (x[:, None] == x[None, :]).float().tril(-1).sum(1) == 0                # first occurrence of its value
    vals = x[first]; return sort_rows(vals[None])[0]
def unique_consecutive_static(x, keep): return x[keep]
def histc(x, bins, lo, hi):
    """torch.histc: bin index floor((x - lo) / (hi - lo) * bins), x == hi in the last bin, outside ignored"""
    w = (hi - lo) / bins; b = torch.floor((x.reshape(-1) - lo) / w); b = torch.where(x.reshape(-1) == hi, torch.full_like(b, bins - 1), b)
    inside = ((x.reshape(-1) >= lo) & (x.reshape(-1) <= hi)).float()
    return ((b[:, None] == torch.arange(bins).float()[None, :]).float() * inside[:, None]).sum(0)
def hist2d(x, bins, rng):
    """torch.histogramdd on (N, 2) with bins x bins over rng in both dims"""
    lo, hi = rng; w = (hi - lo) / bins
    b = torch.floor((x - lo) / w); b = torch.where(x == hi, torch.full_like(b, bins - 1), b)
    inside = ((x >= lo) & (x <= hi)).all(1).float()
    o0 = (b[:, 0:1] == torch.arange(bins).float()[None, :]).float(); o1 = (b[:, 1:2] == torch.arange(bins).float()[None, :]).float()
    return (o0 * inside[:, None]).t() @ o1
def cdist(a, b): return ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1).clamp(min=0).sqrt()
def pdist(x):
    n = x.shape[0]; d = cdist(x, x); iu = triu_pairs(n); return (d * iu).sum((1, 2)) if False else torch.stack([d[i, j] for i, j in [(a, b) for a in range(n) for b in range(a + 1, n)]])
def triu_pairs(n): return [(a, b) for a in range(n) for b in range(a + 1, n)]
def embedding_renorm(w, idx, max_norm, p):
    """torch.embedding_renorm_: the rows in idx scaled to norm <= max_norm (factor max_norm / (norm + 1e-7))"""
    sel = onehot(idx, w.shape[0]).sum(0).clamp(max=1)                              # (V,) 1 on the touched rows
    nrm = w.norm(p, dim=1); sc = torch.where((nrm > max_norm) & (sel > 0), max_norm / (nrm + 1e-7), torch.ones_like(nrm))
    return w * sc[:, None]
def nuclear_norm_4x4(x):
    """sum of the singular values = trace of sqrt(X^T X) via the eigenvalues of the 4x4 SPD X^T X (Jacobi sweeps)"""
    return svd_vals_4x4(x).sum().reshape(1)
def svd_vals_4x4(x):
    """singular values of a 4x4 matrix, descending: sqrt of the eigenvalues of X^T X by cyclic Jacobi rotations"""
    A = x.t() @ x; n = 4
    for _ in range(8):
        for p in range(n):
            for q in range(p + 1, n):
                app, aqq, apq = A[p, p], A[q, q], A[p, q]
                theta = 0.5 * atan2(2 * apq, aqq - app)
                c, s = torch.cos(theta), torch.sin(theta)
                ep = torch.eye(n)[p]; eq = torch.eye(n)[q]
                J = torch.eye(n) + (c - 1) * (ep[:, None] * ep[None, :] + eq[:, None] * eq[None, :]) + s * (ep[:, None] * eq[None, :] - eq[:, None] * ep[None, :])
                A = J.t() @ A @ J
    ev = torch.diagonal(A).clamp(min=0).sqrt()
    return sort_rows(ev[None], descending=True)[0]
def det_4x4(x):
    """determinant by cofactor expansion (no LU pivoting)"""
    def det3(m): return m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]) - m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0]) + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0])
    idx = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
    return sum(((-1) ** j) * x[0, j] * det3(x[1:][:, idx[j]]) for j in range(4))
def inv_4x4(x):
    """inverse by the adjugate: cofactor matrix / det"""
    def det3(m): return m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]) - m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0]) + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0])
    rows = [[a for a in range(4) if a != i] for i in range(4)]
    C = torch.stack([torch.stack([((-1) ** (i + j)) * det3(x[rows[i]][:, rows[j]]) for j in range(4)]) for i in range(4)])
    return C.t() / det_4x4(x)
def lu_nopivot(x):
    """Doolittle LU without pivoting (unit lower L, upper U) -> (L, U)"""
    n = x.shape[0]; U = x.clone(); L = torch.eye(n)
    for k in range(n):
        for i in range(k + 1, n):
            f = U[i, k] / U[k, k]
            L = L + f * (torch.eye(n)[i][:, None] * torch.eye(n)[k][None, :])
            U = torch.cat([U[:i], (U[i] - f * U[k])[None], U[i + 1:]], 0)
    return L, torch.triu(U)
def lu_unpacked_nopivot_ref(x):
    P, L, U = torch.linalg.lu(x); assert torch.equal(P, torch.eye(x.shape[0])), 'pivoting on this matrix'
    return L, U
def matrix_exp_taylor(x, terms=18):
    out = torch.eye(x.shape[0]); term = torch.eye(x.shape[0])
    for k in range(1, terms): term = term @ x / k; out = out + term
    return out
def pinv_tall(x):
    """pseudo-inverse of a full-column-rank tall matrix: (X^T X)^-1 X^T with the 3x3 inverse by the adjugate"""
    A = x.t() @ x
    def det3(m): return m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]) - m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0]) + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0])
    rows = [[a for a in range(3) if a != i] for i in range(3)]
    C = torch.stack([torch.stack([((-1) ** (i + j)) * (A[rows[i]][:, rows[j]][0, 0] * A[rows[i]][:, rows[j]][1, 1] - A[rows[i]][:, rows[j]][0, 1] * A[rows[i]][:, rows[j]][1, 0]) for j in range(3)]) for i in range(3)])
    return (C.t() / det3(A)) @ x.t()
def power_iter_top(A, iters):
    """the largest eigenvalue of an SPD matrix by power iteration from a fixed start"""
    v = torch.ones(A.shape[0]) / math.sqrt(A.shape[0])
    for _ in range(iters): v = A @ v; v = v / v.norm()
    return (v @ A @ v).reshape(1)
# ---- STFT as matmuls (torch.stft lowers with complex types)
def dft_matrix(n):
    k = torch.arange(n).float(); a = 2 * math.pi * k[:, None] * k[None, :n // 2 + 1] / n
    return torch.cat([torch.cos(a), -torch.sin(a)], 1)                              # (n, 2*(n/2+1)): [re | im] of the one-sided DFT
def idft_matrix(n):
    """inverse one-sided DFT: frames (2*(n/2+1)) -> n samples, the redundant bins weighted x2"""
    k = torch.arange(n).float(); f = torch.arange(n // 2 + 1).float(); a = 2 * math.pi * k[None, :] * f[:, None] / n
    w = torch.ones(n // 2 + 1); w[1:-1] = 2.0
    return torch.cat([w[:, None] * torch.cos(a), -w[:, None] * torch.sin(a)], 0) / n   # (2*(n/2+1), n)
def frames(x, n, hop):
    """centered (reflect-padded) frames of x (B, L) -> (B, T, n) with T = L // hop + 1"""
    xp = torch.nn.functional.pad(x, (n // 2, n // 2), mode='reflect'); T = x.shape[1] // hop + 1
    idx = torch.arange(T)[:, None] * hop + torch.arange(n)[None, :]
    return xp[:, idx]
def stft_mat(x, win, dft, n, hop):
    """torch.stft(center, reflect, one-sided) -> (B, [re (F, T) | im (F, T)]) flattened as the reference's view_as_real permute"""
    fr = frames(x, n, hop) * win; z = fr @ dft                                    # (B, T, 2F)
    F = n // 2 + 1; re = z[:, :, :F].transpose(1, 2); im = z[:, :, F:].transpose(1, 2)
    return torch.cat([re.reshape(x.shape[0], -1), im.reshape(x.shape[0], -1)], 1)
def stft_re_im(x, n, hop):
    z = torch.stft(x, n, hop, window=torch.hann_window(n), return_complex=True)
    return torch.cat([z.real.reshape(x.shape[0], -1), z.imag.reshape(x.shape[0], -1)], 1)
def istft_mat(z, win, idft, n, hop, length):
    """torch.istft: inverse DFT per frame, window, overlap-add, divide by the window-square sum, trim the center padding"""
    B = z.shape[0]; F = n // 2 + 1; T = z.shape[1] // (2 * F)
    re = z[:, :F * T].reshape(B, F, T); im = z[:, F * T:].reshape(B, F, T)
    fr = torch.cat([re, im], 1).transpose(1, 2) @ idft * win                       # (B, T, n)
    L = n + hop * (T - 1); idx = torch.arange(T)[:, None] * hop + torch.arange(n)[None, :]
    P = (idx.reshape(-1)[:, None] == torch.arange(L)[None, :]).float()            # (T*n, L) overlap-add
    y = fr.reshape(B, -1) @ P; wss = (win * win).repeat(T) @ P
    y = y / wss.clamp(min=1e-11)
    return y[:, n // 2:n // 2 + length]
def ctc_loss(lp, target):
    """CTC loss (sum) of one sequence: the alpha recursion on the extended label [blank, t1, blank, t2, blank]"""
    T = lp.shape[0]; ext = [0]; [ext.extend([t, 0]) for t in target]; S = len(ext)
    neg = torch.full((S,), -1e30)
    a = neg.clone(); a = torch.cat([lp[0, ext[0]].reshape(1), lp[0, ext[1]].reshape(1), neg[2:]])
    def lae(*xs): m = torch.stack(xs).amax(0); return m + torch.log(torch.stack([torch.exp(v - m) for v in xs]).sum(0))
    for t in range(1, T):
        new = []
        for s in range(S):
            cand = [a[s]] + ([a[s - 1]] if s >= 1 else []) + ([a[s - 2]] if s >= 2 and ext[s] != 0 and ext[s] != ext[s - 2] else [])
            new.append(lae(*cand) + lp[t, ext[s]])
        a = torch.stack(new)
    return -lae(a[S - 1], a[S - 2]).reshape(1)

# ---- special functions (the same series / asymptotics as ../torchintf/intf_special.py); torch-mlir has no
# lowering for aten.lgamma / digamma / polygamma / erfc / erfinv / i0 / igamma / hypot / nextafter
def lgamma(x):
    z = x + 8; acc = torch.zeros_like(x)
    for k in range(8): acc = acc + torch.log(x + k)
    r = 1 / z; r2 = r * r
    return (z - 0.5) * torch.log(z) - z + 0.5 * math.log(2 * math.pi) + r * (1 / 12 - r2 * (1 / 360 - r2 * (1 / 1260 - r2 / 1680))) - acc
def digamma(x):
    z = x + 8; acc = torch.zeros_like(x)
    for k in range(8): acc = acc + 1 / (x + k)
    r = 1 / z; r2 = r * r
    return torch.log(z) - 0.5 * r - r2 * (1 / 12 - r2 * (1 / 120 - r2 * (1 / 252 - r2 / 240))) - acc
def trigamma(x):
    z = x + 8; acc = torch.zeros_like(x)
    for k in range(8): acc = acc + 1 / ((x + k) * (x + k))
    r = 1 / z; r2 = r * r
    return r + 0.5 * r2 + r * r2 * (1 / 6 - r2 * (1 / 30 - r2 * (1 / 42 - r2 / 30))) + acc
def erfinv(x):   # M. Giles (single precision branch)
    w = -torch.log((1 - x) * (1 + x)); small = w < 5; w1 = w - 2.5; w2 = torch.sqrt(w) - 3
    p1 = 2.81022636e-08
    for c in (3.43273939e-07, -3.5233877e-06, -4.39150654e-06, 0.00021858087, -0.00125372503, -0.00417768164, 0.246640727, 1.50140941): p1 = c + p1 * w1
    p2 = -0.000200214257
    for c in (0.000100950558, 0.00134934322, -0.00367342844, 0.00573950773, -0.0076224613, 0.00943887047, 1.00167406, 2.83297682): p2 = c + p2 * w2
    return torch.where(small, p1, p2) * x
def i0(x, K=30):
    t = x * x / 4; acc = torch.ones_like(t)
    for k in range(K, 0, -1): acc = 1 + acc * t / float(k * k)
    return acc
def gammainc(a, x, K=60):
    """regularised lower incomplete gamma P(a, x) by the series x^a e^-x / Gamma(a+1) sum_k x^k / ((a+1)..(a+k)) (x, a in [0.5, 6])"""
    term = torch.ones_like(x); acc = torch.ones_like(x)
    for k in range(1, K + 1): term = term * x / (a + k); acc = acc + term
    return torch.exp(a * torch.log(x) - x - lgamma(a + 1)) * acc
def hypot(x, y): return torch.sqrt(x * x + y * y)

# ---- windows (torch.*_window have no lowering; torch.signal.windows' formulas)
def cosine_window(M, coefs):
    n = torch.arange(M).float(); w = torch.zeros(M)
    for k, a in enumerate(coefs): w = w + ((-1) ** k) * a * torch.cos(2 * math.pi * k * n / M)
    return w
def bartlett_window(M):   # periodic
    n = torch.arange(M).float(); return 1 - (2 * n / M - 1).abs()
def kaiser_window(M, beta):   # periodic
    n = torch.arange(M).float(); r = 2 * n / M - 1
    return i0(beta * torch.sqrt((1 - r * r).clamp(min=0))) / i0(torch.tensor(beta))
# ---- random distributions as inverse CDFs of a pinned uniform draw
def cauchy_icdf(u, median=0.0, sigma=1.0): return median + sigma * torch.tan(math.pi * (u - 0.5))
def exponential_icdf(u, lambd=1.0): return -torch.log(1 - u) / lambd
def geometric_icdf(u, p): return torch.floor(torch.log(1 - u) / math.log(1 - p)) + 1
def log_normal_icdf(u, mean=0.0, std=1.0): return torch.exp(mean + std * math.sqrt(2) * erfinv(2 * u - 1))
def gcd_f(a, b, iters=12):
    """gcd of small positive integers (as floats) by iterated Euclid with a fixed number of steps (a, b < 32)"""
    for _ in range(iters):
        r = a - b * torch.floor(a / b); a, b = b, r
        b = torch.where(b == 0, a, b)     # once the remainder is 0 the pair sticks at (g, g)
    return a
def grid_sample_3d(x, g):
    """torch.grid_sampler_3d (bilinear, zeros padding, align_corners False) on (1, C, D, H, W) with grid (1, Do, Ho, Wo, 3) by one-hot gathers"""
    _, C, D, H, W = x.shape; go = g.shape[1:4]; gg = g.reshape(-1, 3)
    ix = ((gg[:, 0] + 1) * W - 1) / 2; iy = ((gg[:, 1] + 1) * H - 1) / 2; iz = ((gg[:, 2] + 1) * D - 1) / 2
    def corner(i, n):
        lo = torch.floor(i); w = i - lo
        return lo, lo + 1, 1 - w, w
    x0, x1, wx0, wx1 = corner(ix, W); y0, y1, wy0, wy1 = corner(iy, H); z0, z1, wz0, wz1 = corner(iz, D)
    def oh(v, n): return ((v[:, None] == torch.arange(n).float()[None, :]) & (v[:, None] >= 0) & (v[:, None] < n)).float()
    img = x[0].reshape(C, D, H, W); out = 0
    for zc, wz in ((z0, wz0), (z1, wz1)):
        for yc, wy in ((y0, wy0), (y1, wy1)):
            for xc, wx in ((x0, wx0), (x1, wx1)):
                v = torch.einsum('cdhw,pd,ph,pw->cp', img, oh(zc, D), oh(yc, H), oh(xc, W))
                out = out + v * (wz * wy * wx)[None, :]
    return out.reshape(1, C, *go)
def frexp_f(x):
    """torch.frexp: mantissa in [0.5, 1) and integer exponent, from log2 (floor(log2|x|) + 1)"""
    e = torch.floor(log2(x.abs())) + 1; e = torch.where(x == 0, torch.zeros_like(e), e)
    m = x / exp2(e)
    fix = m.abs() >= 1; e = torch.where(fix, e + 1, e); m = torch.where(fix, m / 2, m)   # log2 rounding at exact powers
    fix2 = (m.abs() < 0.5) & (x != 0); e = torch.where(fix2, e - 1, e); m = torch.where(fix2, m * 2, m)
    return m, e
def searchsorted_f(b, x):
    """torch.searchsorted(boundaries, x) (right=False): number of boundaries < x"""
    return (b[None, None, :] < x[:, :, None]).float().sum(-1)
def cov_f(x):
    """torch.cov of (N, M): rows are variables; unbiased"""
    xc = x - x.mean(1, keepdim=True); return xc @ xc.t() / (x.shape[1] - 1)
def corrcoef_f(x):
    c = cov_f(x); d = torch.sqrt(torch.diagonal(c)); return c / (d[:, None] * d[None, :])
def combinations_f(v):
    """torch.combinations(v, 2) of a 5-vector: the 10 (i < j) pairs, a constant gather"""
    n = v.shape[0]; pr = triu_pairs(n); return torch.stack([torch.stack([v[i] for i, j in pr]), torch.stack([v[j] for i, j in pr])], 1)

def nextafter(x, y):
    """the next float32 after x towards y: x + ulp(x) * sign(y - x), ulp from the exponent (frexp)"""
    m, e = frexp_f(x); ulp = exp2(e - 24)
    ulp = torch.where(x == 0, torch.full_like(x, 1.4e-45), ulp)
    down = (y < x) & (m.abs() == 0.5)    # stepping down across a power of two halves the ulp
    ulp = torch.where(down, ulp / 2, ulp)
    return torch.where(y > x, x + ulp, torch.where(y < x, x - ulp, x))


# ---- scatter / index_* (tm_tensor.scatter has no hwacha-mlir lowering) as one-hot products; the index is a constant
def index_add_rows(x, idx, src, alpha=1.0): return x + alpha * onehot(idx, x.shape[0]).t() @ src
def index_copy_rows(x, idx, src):
    o = onehot(idx, x.shape[0]); hit = o.sum(0).clamp(max=1)[:, None]          # (later indices win: none repeat here)
    return x * (1 - hit) + o.t() @ src
def scatter_rows(x, idx, src, reduce=None):
    """torch.scatter(x, 1, idx, src) on (B, N) with idx (B, N): every (b, j) writes src[b, j] to x[b, idx[b, j]];
    reduce None = replace (the last writer wins in torch: unique columns per row here), 'add' = accumulate"""
    B, N = x.shape; o = (idx[:, :, None] == torch.arange(x.shape[1])[None, None, :]).float()   # (B, N, C)
    acc = torch.einsum('bjc,bj->bc', o, src)
    if reduce == 'add': return x + acc
    hit = o.sum(1).clamp(max=1); return x * (1 - hit) + acc
def scatter_amax_rows(x, idx, src):
    B, N = x.shape; o = (idx[:, :, None] == torch.arange(x.shape[1])[None, None, :]).float()
    m = torch.where(o.permute(0, 2, 1) > 0, src[:, None, :], torch.full_like(src[:, None, :].expand(-1, x.shape[1], -1), -1e30)).amax(-1)
    return torch.maximum(x, m)
def scatter_mean_rows_noself(x, idx, src):
    B, N = x.shape; o = (idx[:, :, None] == torch.arange(x.shape[1])[None, None, :]).float()
    cnt = o.sum(1); acc = torch.einsum('bjc,bj->bc', o, src)
    return torch.where(cnt > 0, acc / cnt.clamp(min=1), x)
def scatter_cols(x, idx, src):
    """torch.scatter_add(x, 0, idx, src): every (i, j) adds src[i, j] to x[idx[i, j], j]"""
    o = (idx[:, :, None] == torch.arange(x.shape[0])[None, None, :]).float()    # (I, J, R)
    return x + torch.einsum('ijr,ij->rj', o, src)
def index_put_acc(x, i, j, v):
    B, N = x.shape; flat_idx = i * N + j
    return x + (onehot(flat_idx, B * N).t() @ v).reshape(B, N)
def alpha_dropout_affine(x, p):
    """torch.alpha_dropout under seed 0 is an affine map a*x + b per element (a, b from the pinned mask): recovered from two probes"""
    torch.manual_seed(0); y0 = torch.alpha_dropout(torch.zeros_like(x), p, True)
    torch.manual_seed(0); y1 = torch.alpha_dropout(torch.ones_like(x), p, True)
    return dict(a=y1 - y0, b=y0)
def rrelu_slopes(x, lo, hi):
    """the per-element negative slopes torch.rrelu drew under seed 0 (rrelu(x) / x on the negative entries)"""
    torch.manual_seed(0); y = torch.rrelu(x, lo, hi, True)
    return torch.where(x < 0, y / torch.where(x < 0, x, torch.ones_like(x)), torch.ones_like(x))
