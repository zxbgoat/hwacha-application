"""torch.special cases (docs.pytorch.org/docs/2.14/special.html) for export_intf.py. torch-mlir lowers
the elementary ones (erf, exp, log, log1p, expm1, sigmoid, round, sin, softmax); the Bessel / Airy /
gamma / zeta functions, erfinv and the orthogonal polynomials have no lowering, so their exported
graphs evaluate them with fixed-length series, asymptotic expansions or recurrences on a restricted
input range (the reference is the genuine call):
  J0, J1, I0, I1        power series in t = x^2/4 by Horner (25 terms), x in [0.5, 4] / [0.5, 3]
  Y0, Y1, K0, K1        the log-series (harmonic numbers) with J / I, same range
  Airy Ai               Maclaurin series (20 terms), x in [-2, 2]
  lgamma, digamma,      shift x -> x + 8 (log / reciprocal sums), then the Stirling / asymptotic
  trigamma              expansions, x in [0.5, 4]; multigammaln = sum of lgamma
  gammainc              series x^a e^-x / Gamma(a+1) sum_k x^k / ((a+1)...(a+k)), a = 2.5, 40 terms
  Hurwitz zeta          Euler-Maclaurin: 10 direct terms + tail integral + 3 Bernoulli terms
  erfinv, ndtri         Giles' single-precision rational approximation, |x| <= 0.9
  erfc, erfcx, log_ndtr from erf on [-2, 2] (no cancellation issue at these magnitudes)
  polynomials           three-term recurrences with n = 5"""
import math, torch, torch.special as S
EG = 0.5772156649015329
def add_special(case, rcase, R):
    torch.manual_seed(2)
    def U(lo, hi, *shape): return torch.rand(*shape) * (hi - lo) + lo
    x4 = U(0.5, 4, 4, 16); x3 = U(0.5, 3, 4, 16); xa = U(-2, 2, 4, 16); u = U(-1, 1, 4, 16); p = U(-0.9, 0.9, 4, 16); s01 = U(0, 1, 4, 16)
    # ---- direct exports (elementary functions torch-mlir lowers)
    case('entr', lambda s, x: S.entr(x), x3); case('erf', lambda s, x: S.erf(x), xa); case('expit', lambda s, x: S.expit(x), xa)
    case('expm1', lambda s, x: S.expm1(x), xa); case('log1p', lambda s, x: S.log1p(x), x3); case('logit', lambda s, x: S.logit(x), U(0.05, 0.95, 4, 16))
    case('log_softmax', lambda s, x: S.log_softmax(x, 1), xa); case('softmax', lambda s, x: S.softmax(x, 1), xa); case('logsumexp', lambda s, x: S.logsumexp(x, 1), xa)
    case('ndtr', lambda s, x: S.ndtr(x), xa); case('round', lambda s, x: S.round(x), xa * 3); case('sinc', lambda s, x: S.sinc(x), xa)
    case('xlogy', lambda s, x: S.xlogy(x, s.y), x3, y=U(0.5, 3, 4, 16)); case('xlog1py', lambda s, x: S.xlog1py(x, s.y), x3, y=U(0.5, 3, 4, 16))
    rcase('exp2', lambda s, x: torch.exp(x * math.log(2.0)), lambda s, x: S.exp2(x), xa)   # powf(2, x) gives hwacha-mlir no kernel
    # ---- erf family
    rcase('erfc', lambda s, x: 1 - torch.erf(x), lambda s, x: S.erfc(x), xa)
    rcase('erfcx', lambda s, x: torch.exp(x * x) * (1 - torch.erf(x)), lambda s, x: S.erfcx(x), U(0, 2, 4, 16))
    rcase('log_ndtr', lambda s, x: torch.log(0.5 * (1 + torch.erf(x / math.sqrt(2)))), lambda s, x: S.log_ndtr(x), U(-2, 3, 4, 16))
    def erfinv(x):   # M. Giles, "Approximating the erfinv function" (single precision branch)
        w = -torch.log((1 - x) * (1 + x)); small = w < 5
        w1 = w - 2.5; w2 = torch.sqrt(w) - 3
        p1 = 2.81022636e-08
        for c in (3.43273939e-07, -3.5233877e-06, -4.39150654e-06, 0.00021858087, -0.00125372503, -0.00417768164, 0.246640727, 1.50140941): p1 = c + p1 * w1
        p2 = -0.000200214257
        for c in (0.000100950558, 0.00134934322, -0.00367342844, 0.00573950773, -0.0076224613, 0.00943887047, 1.00167406, 2.83297682): p2 = c + p2 * w2
        return torch.where(small, p1, p2) * x
    rcase('erfinv', lambda s, x: erfinv(x), lambda s, x: S.erfinv(x), p)
    rcase('ndtri', lambda s, x: math.sqrt(2) * erfinv(2 * x - 1), lambda s, x: S.ndtri(x), U(0.05, 0.95, 4, 16))
    # ---- gamma family (x in [0.5, 4]: shift by 8, then the asymptotic expansions)
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
    rcase('gammaln', lambda s, x: lgamma(x), lambda s, x: S.gammaln(x), x4)
    rcase('digamma', lambda s, x: digamma(x), lambda s, x: S.digamma(x), x4)
    rcase('psi', lambda s, x: digamma(x), lambda s, x: S.psi(x), x4)
    rcase('polygamma', lambda s, x: trigamma(x), lambda s, x: S.polygamma(1, x), x4)
    rcase('multigammaln', lambda s, x: 1.5 * math.log(math.pi) + lgamma(x) + lgamma(x - 0.5) + lgamma(x - 1), lambda s, x: S.multigammaln(x, 3), U(2.5, 5.5, 4, 16))
    a = 2.5; ga1 = math.gamma(a + 1)
    def gammainc(x, K=40):
        term = torch.ones_like(x); acc = torch.ones_like(x)
        for k in range(1, K + 1): term = term * x / (a + k); acc = acc + term
        return torch.exp(a * torch.log(x) - x) / ga1 * acc
    rcase('gammainc', lambda s, x: gammainc(x), lambda s, x: S.gammainc(s.a, x), U(0.5, 5, 4, 16), a=torch.full((4, 16), a))
    rcase('gammaincc', lambda s, x: 1 - gammainc(x), lambda s, x: S.gammaincc(s.a, x), U(0.5, 5, 4, 16), a=torch.full((4, 16), a))
    zx = 3.5
    def hzeta(q, N=10):   # Hurwitz zeta(zx, q), Euler-Maclaurin
        acc = torch.zeros_like(q)
        for k in range(N): acc = acc + torch.exp(-zx * torch.log(q + k))
        b = q + N; lb = torch.log(b)
        return acc + torch.exp((1 - zx) * lb) / (zx - 1) + 0.5 * torch.exp(-zx * lb) + zx / 12 * torch.exp(-(zx + 1) * lb) - zx * (zx + 1) * (zx + 2) / 720 * torch.exp(-(zx + 3) * lb)
    rcase('zeta', lambda s, x: hzeta(x), lambda s, x: S.zeta(s.zx, x), U(1, 5, 4, 16), zx=torch.full((4, 16), zx))
    # ---- Bessel and Airy (power series in t = x^2 / 4)
    K = 25
    H = [0.0]; [H.append(H[-1] + 1 / k) for k in range(1, K + 2)]                      # harmonic numbers
    def series(t, coef):   # sum_k coef[k] t^k by Horner
        acc = torch.full_like(t, coef[K])
        for k in range(K - 1, -1, -1): acc = coef[k] + acc * t
        return acc
    fk = [math.factorial(k) for k in range(K + 2)]
    cJ0 = [(-1) ** k / fk[k] ** 2 for k in range(K + 1)]; cI0 = [1 / fk[k] ** 2 for k in range(K + 1)]
    cJ1 = [(-1) ** k / (fk[k] * fk[k + 1]) for k in range(K + 1)]; cI1 = [1 / (fk[k] * fk[k + 1]) for k in range(K + 1)]
    cY0 = [(-1) ** (k + 1) * H[k] / fk[k] ** 2 for k in range(K + 1)]; cK0 = [H[k] / fk[k] ** 2 for k in range(K + 1)]
    cY1 = [(-1) ** k * (H[k] + H[k + 1]) / (fk[k] * fk[k + 1]) for k in range(K + 1)]; cK1 = [(H[k] + H[k + 1]) / (fk[k] * fk[k + 1]) for k in range(K + 1)]
    def J0(x): return series(x * x / 4, cJ0)
    def J1(x): return x / 2 * series(x * x / 4, cJ1)
    def I0(x): return series(x * x / 4, cI0)
    def I1(x): return x / 2 * series(x * x / 4, cI1)
    def Y0(x): return 2 / math.pi * ((torch.log(x / 2) + EG) * J0(x) + series(x * x / 4, cY0))
    def Y1(x): return 2 / math.pi * ((torch.log(x / 2) + EG) * J1(x) - 1 / x) - x / (2 * math.pi) * series(x * x / 4, cY1)
    def K0(x): return -(torch.log(x / 2) + EG) * I0(x) + series(x * x / 4, cK0)
    def K1(x): return 1 / x + (torch.log(x / 2) + EG) * I1(x) - x / 4 * series(x * x / 4, cK1)
    rcase('bessel_j0', lambda s, x: J0(x), lambda s, x: S.bessel_j0(x), x4); rcase('bessel_j1', lambda s, x: J1(x), lambda s, x: S.bessel_j1(x), x4)
    rcase('bessel_y0', lambda s, x: Y0(x), lambda s, x: S.bessel_y0(x), x4); rcase('bessel_y1', lambda s, x: Y1(x), lambda s, x: S.bessel_y1(x), x4)
    rcase('modified_bessel_i0', lambda s, x: I0(x), lambda s, x: S.modified_bessel_i0(x), x3); rcase('modified_bessel_i1', lambda s, x: I1(x), lambda s, x: S.modified_bessel_i1(x), x3)
    rcase('modified_bessel_k0', lambda s, x: K0(x), lambda s, x: S.modified_bessel_k0(x), x3); rcase('modified_bessel_k1', lambda s, x: K1(x), lambda s, x: S.modified_bessel_k1(x), x3)
    rcase('i0', lambda s, x: I0(x), lambda s, x: S.i0(x), x3); rcase('i1', lambda s, x: I1(x), lambda s, x: S.i1(x), x3)
    rcase('i0e', lambda s, x: torch.exp(-x) * I0(x), lambda s, x: S.i0e(x), x3); rcase('i1e', lambda s, x: torch.exp(-x) * I1(x), lambda s, x: S.i1e(x), x3)
    rcase('scaled_modified_bessel_k0', lambda s, x: torch.exp(x) * K0(x), lambda s, x: S.scaled_modified_bessel_k0(x), x3)
    rcase('scaled_modified_bessel_k1', lambda s, x: torch.exp(x) * K1(x), lambda s, x: S.scaled_modified_bessel_k1(x), x3)
    rcase('spherical_bessel_j0', lambda s, x: torch.sin(x) / x, lambda s, x: S.spherical_bessel_j0(x), x4)
    def airy(x, K=20):   # Ai = c1 f - c2 g; f, g the two Maclaurin series
        x3 = x * x * x; f = torch.ones_like(x); g = x.clone(); tf = torch.ones_like(x); tg = x.clone()
        for k in range(1, K + 1):
            tf = tf * x3 / ((3 * k - 1) * (3 * k)); tg = tg * x3 / ((3 * k) * (3 * k + 1)); f = f + tf; g = g + tg
        return 0.3550280538878172 * f - 0.2588194037928068 * g
    rcase('airy_ai', lambda s, x: airy(x), lambda s, x: S.airy_ai(x), xa)
    # ---- orthogonal polynomials, n = 5, three-term recurrences
    nn = 5
    def cheb(x, p1):   # T / U / V / W share the recurrence, differ in P1
        p0 = torch.ones_like(x); a, b = p0, p1
        for _ in range(nn - 1): a, b = b, 2 * x * b - a
        return b
    polys = {'chebyshev_polynomial_t': lambda x: cheb(x, x), 'chebyshev_polynomial_u': lambda x: cheb(x, 2 * x), 'chebyshev_polynomial_v': lambda x: cheb(x, 2 * x - 1), 'chebyshev_polynomial_w': lambda x: cheb(x, 2 * x + 1)}
    for nm, f in polys.items():
        rcase(nm, (lambda f: lambda s, x: f(x))(f), (lambda nm: lambda s, x: getattr(S, nm)(x, s.n))(nm), u, n=torch.full((4, 16), float(nn)))
        rcase('shifted_' + nm, (lambda f: lambda s, x: f(2 * x - 1))(f), (lambda nm: lambda s, x: getattr(S, 'shifted_' + nm)(x, s.n))(nm), s01, n=torch.full((4, 16), float(nn)))
    def hermite_h(x):
        a, b = torch.ones_like(x), 2 * x
        for k in range(1, nn): a, b = b, 2 * x * b - 2 * k * a
        return b
    def hermite_he(x):
        a, b = torch.ones_like(x), x
        for k in range(1, nn): a, b = b, x * b - k * a
        return b
    def laguerre(x):
        a, b = torch.ones_like(x), 1 - x
        for k in range(1, nn): a, b = b, ((2 * k + 1 - x) * b - k * a) / (k + 1)
        return b
    def legendre(x):
        a, b = torch.ones_like(x), x
        for k in range(1, nn): a, b = b, ((2 * k + 1) * x * b - k * a) / (k + 1)
        return b
    rcase('hermite_polynomial_h', lambda s, x: hermite_h(x), lambda s, x: S.hermite_polynomial_h(x, s.n), u, n=torch.full((4, 16), float(nn)))
    rcase('hermite_polynomial_he', lambda s, x: hermite_he(x), lambda s, x: S.hermite_polynomial_he(x, s.n), u, n=torch.full((4, 16), float(nn)))
    rcase('laguerre_polynomial_l', lambda s, x: laguerre(x), lambda s, x: S.laguerre_polynomial_l(x, s.n), x3, n=torch.full((4, 16), float(nn)))
    rcase('legendre_polynomial_p', lambda s, x: legendre(x), lambda s, x: S.legendre_polynomial_p(x, s.n), u, n=torch.full((4, 16), float(nn)))
