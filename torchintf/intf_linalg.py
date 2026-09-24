"""torch.linalg cases (docs.pytorch.org/docs/2.14/linalg.html) for export_intf.py. torch-mlir lowers
only the products / norms / diagonal / det directly; every factorization and solver is a fixed-size
composition of linalg-friendly ops on 4x4 matrices (the reference is the genuine torch.linalg call):
  inverse            Newton-Schulz X <- X (2I - A X), X0 = A^T / (|A|_1 |A|_inf), 30 iterations
  LU (P = I)         Gauss transforms without pivoting; the inputs are diagonally dominant, so LAPACK
                     does not pivot either (asserted at export)
  triangular solve   (I + N)^-1 = I - N + N^2 - N^3 exactly (N nilpotent), U = D (I + M)
  cholesky / LDL     from the unpivoted LU of an SPD matrix: A = L D L^T, D = diag(U)
  eigh (symmetric)   cyclic Jacobi, 10 sweeps of the 6 rotations, eigenvalues sorted by a rank
                     permutation, eigenvector signs fixed by the first component (reference likewise)
  svd                Jacobi on A^T A: V, sigma = sqrt(w) descending, U = A V / sigma
  qr                 modified Gram-Schmidt; the reference's signs are normalized to diag(R) > 0
  matrix_exp         scaling (A/4) and squaring, Taylor by Horner (15 terms)
  concatenation      [A | B] = A @ [I 0] + B @ [0 I] (constant buffers): no tensor.concat"""
import torch, torch.linalg as L
def add_linalg(case, rcase, R):
    n = 4; I4 = torch.eye(n); ar = torch.arange(n).float()
    below = torch.stack([(torch.arange(n) > k).float() for k in range(n)])          # below[k][i] = i > k
    E1 = torch.cat([I4, torch.zeros(n, n)], 1); E2 = torch.cat([torch.zeros(n, n), I4], 1)   # [A | B] = A @ E1 + B @ E2
    def hcat(A, B, E1, E2): return A @ E1 + B @ E2
    def outer(a, b): return a[:, None] * b[None, :]
    def diagv(A, I): return (A * I).sum(-1)                                          # diagonal as a vector
    def diagm(d, I): return I * d[:, None]
    def sgn(x): return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
    # ---- building blocks (s: the module, for its buffers)
    def ns_inv(s, A, iters=30):
        X = A.transpose(-1, -2) / (A.abs().sum(-2).max() * A.abs().sum(-1).max())
        for _ in range(iters): X = X @ (2 * s.I - A @ X)
        return X
    def lu_np(s, A):
        Lm = s.I; U = A
        for k in range(n - 1):
            col = U @ s.I[k]; piv = (col * s.I[k]).sum()
            l = col * s.below[k] / piv
            U = (s.I - outer(l, s.I[k])) @ U; Lm = Lm + outer(l, s.I[k])
        return Lm, U
    def unit_lower_inv(s, Lm):
        N = Lm - s.I; N2 = N @ N; return s.I - N + N2 - N2 @ N
    def upper_inv(s, U):
        d = diagv(U, s.I); Dinv = diagm(1 / d, s.I); M = Dinv @ (U - diagm(d, s.I)); M2 = M @ M
        return (s.I - M + M2 - M2 @ M) @ Dinv
    def jacobi(s, A, sweeps=10):
        V = s.I
        for _ in range(sweeps):
            for p in range(n):
                for q in range(p + 1, n):
                    ep, eq = s.I[p], s.I[q]
                    Epp, Eqq, Epq, Eqp = outer(ep, ep), outer(eq, eq), outer(ep, eq), outer(eq, ep)
                    app, aqq, apq = (A * Epp).sum(), (A * Eqq).sum(), (A * Epq).sum()
                    d = aqq - app
                    t = 2 * apq * sgn(d) / (d.abs() + torch.sqrt(d * d + 4 * apq * apq))
                    c = 1 / torch.sqrt(1 + t * t); sn = t * c
                    J = s.I + (c - 1) * (Epp + Eqq) + sn * Epq - sn * Eqp
                    A = J.transpose(-1, -2) @ A @ J; V = V @ J
        return diagv(A, s.I), V
    def sort_perm(s, w, descending=False):
        """P with P @ w sorted: P[i, j] = (rank_j == i), rank_j = #{m : w_m < w_j} (or >)"""
        cmp = (w[None, :] > w[:, None]) if descending else (w[None, :] < w[:, None])   # cmp[j, m]
        rank = cmp.float().sum(-1)
        return (rank[None, :] == s.ar[:, None]).float()
    def fix_sign(V): return V * sgn(V[0])[None, :]
    def eigh_sorted(s, A):
        w, V = jacobi(s, A); P = sort_perm(s, w)
        return P @ w, fix_sign(V @ P.transpose(-1, -2))
    def svd_parts(s, A):
        w, V = jacobi(s, A.transpose(-1, -2) @ A); P = sort_perm(s, w, descending=True)
        sig = torch.sqrt(torch.clamp(P @ w, min=0)); V = fix_sign(V @ P.transpose(-1, -2))
        return A @ V / sig[None, :], sig, V.transpose(-1, -2)
    def ref_eigh(A):
        w, V = L.eigh(A); return w, fix_sign(V)
    def ref_svd(A):
        U, S, Vh = L.svd(A); sg = sgn(Vh[:, 0]); return U * sg[None, :], S, Vh * sg[:, None]
    # ---- inputs: diagonally dominant (no pivoting, well conditioned), fixed seed
    torch.manual_seed(1)
    A = torch.randn(n, n) * 0.5 + 4 * I4; S = A @ A.T; B = torch.randn(n, 2); T6 = torch.randn(6, n); v3 = torch.randn(n, 3)
    LU_ref, piv = L.lu_factor(A); assert torch.equal(piv, torch.arange(1, n + 1, dtype=piv.dtype)), 'LAPACK pivoted'
    LD_ref, ldl_piv = L.ldl_factor(S); assert torch.equal(ldl_piv, torch.arange(1, n + 1, dtype=ldl_piv.dtype)), 'sytrf pivoted'
    common = dict(I=I4, ar=ar, below=below, E1=E1, E2=E2)
    # ---- matrix properties
    rcase('norm', lambda s, x: torch.sqrt((x * x).sum()).reshape(1), lambda s, x: L.norm(x).reshape(1), A)
    # (vector_norm / matrix_norm export as pow-based generics that hwacha-mlir turns into no kernel: same sqrt(sum x^2))
    rcase('vector_norm', lambda s, x: torch.sqrt((x * x).sum()).reshape(1), lambda s, x: L.vector_norm(x).reshape(1), A)
    rcase('matrix_norm', lambda s, x: torch.sqrt((x * x).sum()).reshape(1), lambda s, x: L.matrix_norm(x).reshape(1), A)
    case('diagonal', lambda s, x: L.diagonal(x), A)
    rcase('det', lambda s, x: torch.prod(diagv(lu_np(s, x)[1], s.I)).reshape(1), lambda s, x: L.det(x).reshape(1), A, **common)
    rcase('slogdet', lambda s, x: (lambda d: torch.prod(sgn(d)) * s.I[0][:2] + torch.log(d.abs()).sum() * s.I[1][:2])(diagv(lu_np(s, x)[1], s.I)),
          lambda s, x: torch.stack(L.slogdet(x)), A, **common)
    rcase('cond', lambda s, x: (lambda sig: sig[0] / sig[3])(svd_parts(s, x)[1]).reshape(1), lambda s, x: L.cond(x).reshape(1), A, **common)
    A3 = torch.randn(6, 3) @ torch.randn(3, n)   # rank 3
    rcase('matrix_rank', lambda s, x: (lambda sig: (sig > 1e-3 * sig[0]).float().sum())(svd_parts(s, x)[1]).reshape(1),
          lambda s, x: L.matrix_rank(x, rtol=1e-3).float().reshape(1), A3, **common)
    # ---- decompositions
    rcase('cholesky', lambda s, x: (lambda LU: LU[0] * torch.sqrt(diagv(LU[1], s.I))[None, :])(lu_np(s, x)), lambda s, x: L.cholesky(x), S, **common)
    def gram_schmidt(s, x):
        qs = []
        for j in range(n):
            a = x @ s.I[j]
            for q in qs: a = a - (q * a).sum() * q
            qs.append(a / torch.sqrt((a * a).sum()))
        Q = sum(outer(q, s.I[j]) for j, q in enumerate(qs)); return Q, Q.transpose(-1, -2) @ x
    def ref_qr(x):
        Q, Rm = L.qr(x); sg = sgn(diagv(Rm, I4)); return Q * sg[None, :], Rm * sg[:, None]
    rcase('qr', lambda s, x: hcat(*gram_schmidt(s, x), s.E1, s.E2), lambda s, x: hcat(*ref_qr(x), E1, E2), A, **common)
    rcase('lu', lambda s, x: hcat(*lu_np(s, x), s.E1, s.E2), lambda s, x: (lambda P, Lm, U: hcat(P.transpose(-1, -2) @ Lm, U, E1, E2))(*L.lu(x)), A, **common)
    rcase('lu_factor', lambda s, x: (lambda Lm, U: Lm - s.I + U)(*lu_np(s, x)), lambda s, x: L.lu_factor(x)[0], A, **common)
    rcase('eigh', lambda s, x: (lambda w, V: hcat(V, w[:, None], s.E45, s.E15))(*eigh_sorted(s, x)), lambda s, x: (lambda w, V: hcat(V, w[:, None], E45, E15))(*ref_eigh(x)),
          S, E45=torch.cat([I4, torch.zeros(n, 1)], 1), E15=torch.cat([torch.zeros(1, n), torch.ones(1, 1)], 1), **common)
    E45 = torch.cat([I4, torch.zeros(n, 1)], 1); E15 = torch.cat([torch.zeros(1, n), torch.ones(1, 1)], 1)
    rcase('eigvalsh', lambda s, x: eigh_sorted(s, x)[0], lambda s, x: L.eigvalsh(x), S, **common)
    def ref_eig(x):   # symmetric input: real eigenvalues, real eigenvectors up to a phase; sorted ascending, signs by the first component
        w, V = L.eig(x); o = torch.argsort(w.real); V = V[:, o].real; return w.real[o], fix_sign(V)
    rcase('eig', lambda s, x: (lambda w, V: hcat(V, w[:, None], s.E45, s.E15))(*eigh_sorted(s, x)), lambda s, x: (lambda w, V: hcat(V, w[:, None], E45, E15))(*ref_eig(x)),
          S, E45=E45, E15=E15, **common)
    rcase('eigvals', lambda s, x: eigh_sorted(s, x)[0], lambda s, x: torch.sort(L.eigvals(x).real)[0], S, **common)
    E49 = torch.cat([I4, torch.zeros(n, 5)], 1); E19 = torch.zeros(1, 9); E19[0, 4] = 1; E49b = torch.cat([torch.zeros(n, 5), I4], 1)
    rcase('svd', lambda s, x: (lambda U, sig, Vh: U @ s.E49 + sig[:, None] @ s.E19 + Vh @ s.E49b)(*svd_parts(s, x)),
          lambda s, x: (lambda U, sig, Vh: U @ E49 + sig[:, None] @ E19 + Vh @ E49b)(*ref_svd(x)), A, E49=E49, E19=E19, E49b=E49b, **common)
    rcase('svdvals', lambda s, x: svd_parts(s, x)[1], lambda s, x: L.svdvals(x), A, **common)
    # ---- solvers and inverses
    rcase('solve', lambda s, x: ns_inv(s, x) @ s.B, lambda s, x: L.solve(x, s.B), A, B=B, **common)
    rcase('solve_triangular', lambda s, x: upper_inv(s, x) @ s.B, lambda s, x: L.solve_triangular(x, s.B, upper=True), A.triu(), B=B, **common)
    rcase('lu_solve', lambda s, x: upper_inv(s, s.LU.triu()) @ unit_lower_inv(s, s.LU.tril(-1) + s.I) @ x, lambda s, x: L.lu_solve(s.LU, s.piv, x), B, LU=LU_ref, piv=piv, **common)
    rcase('lstsq', lambda s, x: ns_inv(s, x.transpose(-1, -2) @ x) @ x.transpose(-1, -2) @ s.B6, lambda s, x: L.lstsq(x, s.B6).solution, T6, B6=T6 @ v3, **common)
    rcase('inv', lambda s, x: ns_inv(s, x), lambda s, x: L.inv(x), A, **common)
    rcase('pinv', lambda s, x: ns_inv(s, x.transpose(-1, -2) @ x) @ x.transpose(-1, -2), lambda s, x: L.pinv(x), T6, **common)
    # ---- matrix functions
    def mexp(s, x):
        Y = x / 4; T = s.I
        for k in range(15, 0, -1): T = s.I + Y @ T / k
        return T @ T @ (T @ T)
    rcase('matrix_exp', mexp, lambda s, x: L.matrix_exp(x), A * 0.2, **common)
    case('matrix_power', lambda s, x: L.matrix_power(x, 3), A)
    # ---- matrix products
    i1, i2 = torch.tensor([1, 2, 0]), torch.tensor([2, 0, 1])
    rcase('cross', lambda s, x: x.index_select(1, s.i1) * s.y.index_select(1, s.i2) - x.index_select(1, s.i2) * s.y.index_select(1, s.i1),
          lambda s, x: L.cross(x, s.y), v3, y=torch.randn(n, 3), i1=i1, i2=i2)
    case('matmul', lambda s, x: L.matmul(x, s.y), A, y=torch.randn(n, n))
    case('vecdot', lambda s, x: L.vecdot(x, s.y), A, y=torch.randn(n, n))
    case('multi_dot', lambda s, x: L.multi_dot([x, s.y, s.z]), A, y=torch.randn(n, 3), z=torch.randn(3, n))
    I6 = torch.eye(6); below6 = torch.stack([(torch.arange(6) > k).float() for k in range(3)]); E63 = I6[:, :3].clone()
    def hh(s, x):
        H = s.I6
        for i in range(3):
            vv = (x @ s.I6[i][:3]) * s.below6[i] + s.I6[i]
            H = H @ (s.I6 - s.tau[i] * outer(vv, vv))
        return H @ s.E63
    rcase('householder_product', hh, lambda s, x: L.householder_product(x, s.tau), T6[:, :3], tau=torch.tensor([0.5, 0.7, 0.3]), I6=I6, below6=below6, E63=E63)
    rcase('tensorinv', lambda s, x: ns_inv(s, x.reshape(n, n)).reshape(2, 2, n), lambda s, x: L.tensorinv(x, ind=1), A.reshape(n, 2, 2), **common)
    rcase('tensorsolve', lambda s, x: ns_inv(s, x.reshape(n, n)) @ s.b, lambda s, x: L.tensorsolve(x, s.b.reshape(2, 2)), A.reshape(2, 2, n), b=B[:, 0].clone(), **common)
    rcase('vander', lambda s, x: sum(outer(x ** k, s.I[k]) for k in range(n)), lambda s, x: L.vander(x), A[0].clone(), **common)
    # ---- experimental (result, info): info is 0 for these inputs (asserted), the result is compared
    for nm, S_or_A in (('cholesky_ex', S), ('inv_ex', A), ('solve_ex', A), ('lu_factor_ex', A), ('ldl_factor_ex', S)):
        assert int(getattr(L, nm)(S_or_A, B)[1] if nm == 'solve_ex' else getattr(L, nm)(S_or_A)[-1]) == 0
    rcase('cholesky_ex', lambda s, x: (lambda LU: LU[0] * torch.sqrt(diagv(LU[1], s.I))[None, :])(lu_np(s, x)), lambda s, x: L.cholesky_ex(x)[0], S, **common)
    rcase('inv_ex', lambda s, x: ns_inv(s, x), lambda s, x: L.inv_ex(x)[0], A, **common)
    rcase('solve_ex', lambda s, x: ns_inv(s, x) @ s.B, lambda s, x: L.solve_ex(x, s.B)[0], A, B=B, **common)
    rcase('lu_factor_ex', lambda s, x: (lambda Lm, U: Lm - s.I + U)(*lu_np(s, x)), lambda s, x: L.lu_factor_ex(x)[0], A, **common)
    def ldl(s, x): Lm, U = lu_np(s, x); return (Lm - s.I) + diagm(diagv(U, s.I), s.I)
    rcase('ldl_factor', ldl, lambda s, x: L.ldl_factor(x)[0].tril(), S, **common)
    rcase('ldl_factor_ex', ldl, lambda s, x: L.ldl_factor_ex(x)[0].tril(), S, **common)
    rcase('ldl_solve', lambda s, x: (lambda Lm: unit_lower_inv(s, Lm).transpose(-1, -2) @ (diagm(1 / diagv(s.LD, s.I), s.I) @ (unit_lower_inv(s, Lm) @ x)))(s.LD.tril(-1) + s.I),
          lambda s, x: L.ldl_solve(s.LD, s.piv, x), B, LD=LD_ref, piv=ldl_piv, **common)
