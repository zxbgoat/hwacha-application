"""Tensor re-implementations used by export_taf.py for the torchaudio.transforms parts torch-mlir
cannot lower: complex spectrograms are real tensors with a trailing (re, im) axis (torch.stft itself
exports through view_as_real; istft, angle, conj, complex abs / solve and fft_irfft do not), the
inverse STFT is an inverse-DFT matmul with an overlap-add matmul and torch.istft's window-sum-square
normalisation, IIR filters are impulse-response matmuls, cumulative sums are triangular matmuls."""
import math, torch, torch.nn as nn, torch.nn.functional as NF, torchaudio.functional as AF

def hann(n): return torch.hann_window(n)
class STFT(nn.Module):
    """torch.stft(x, n_fft, hop, window=hann(n_fft), center=True, pad_mode='reflect', normalized=False, onesided=True)
    for a fixed signal length, as a real tensor (..., F, T, 2): torch-mlir's own stft lowering leaves complex
    types and dynamic shapes that hwacha-mlir does not take, so the frames are gathered with a constant
    index and multiplied by the windowed DFT matrix"""
    def __init__(s, n_fft, hop, length):
        super().__init__(); N, F_ = n_fft, n_fft // 2 + 1; s.N, s.F = N, F_
        T = 1 + length // hop
        n = torch.arange(N, dtype=torch.float64); w = 0.5 - 0.5 * torch.cos(2 * math.pi * n / N)
        ang = 2 * math.pi * torch.outer(n, torch.arange(F_, dtype=torch.float64)) / N
        s.register_buffer('frame_idx', (torch.arange(T)[:, None] * hop + torch.arange(N)[None, :]).reshape(-1))
        s.register_buffer('dft', torch.cat([w[:, None] * torch.cos(ang), -w[:, None] * torch.sin(ang)], 1).float())   # (N, 2F)
    def forward(s, x):   # (..., L) -> (..., F, T, 2)
        lead = x.shape[:-1]; xp = NF.pad(x.reshape(-1, x.shape[-1]), (s.N // 2, s.N // 2), mode='reflect')
        fr = xp.index_select(1, s.frame_idx).reshape(xp.shape[0], -1, s.N)
        z = (fr @ s.dft).reshape(xp.shape[0], -1, 2, s.F).permute(0, 3, 1, 2)
        return z.reshape(*lead, s.F, z.shape[-2], 2)
def power(z): return z[..., 0] ** 2 + z[..., 1] ** 2
def cabs(z): return torch.sqrt(power(z))
def cmul(a, b): return torch.stack([a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1], a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]], -1)
def cconj(a): return torch.stack([a[..., 0], -a[..., 1]], -1)
def cdiv(a, b): d = power(b); n = cmul(a, cconj(b)); return n / d[..., None]
def cscale(a, r): return a * r[..., None]
def atan2(y, x):
    ax = torch.where(x == 0, torch.full_like(x, 1e-30), x); a = torch.atan(y / ax)
    return torch.where(x < 0, a + torch.where(y >= 0, torch.full_like(a, math.pi), torch.full_like(a, -math.pi)), a)
def cangle(z): return atan2(z[..., 1], z[..., 0])
def polar(mag, ph): return torch.stack([mag * torch.cos(ph), mag * torch.sin(ph)], -1)

class ISTFT(nn.Module):
    """torch.istft(z, n_fft, hop, window=hann(n_fft), center=True, normalized=False, length=length) for a
    fixed number of frames, as matmuls: frames = z @ idft (the c_f-weighted inverse DFT), times the window,
    overlap-added by a 0/1 matrix, divided by the window-sum-square, trimmed by n_fft / 2 and to length"""
    def __init__(s, n_fft, hop, frames, length):
        super().__init__(); N, F_ = n_fft, n_fft // 2 + 1; s.N, s.hop, s.T, s.length = N, hop, frames, length
        n = torch.arange(N, dtype=torch.float64); w = 0.5 - 0.5 * torch.cos(2 * math.pi * n / N)
        ang = 2 * math.pi * torch.outer(torch.arange(F_, dtype=torch.float64), n) / N
        c = torch.full((F_,), 2.0, dtype=torch.float64); c[0] = 1; c[-1] = 1
        s.register_buffer('idft', (torch.cat([c[:, None] * torch.cos(ang), -c[:, None] * torch.sin(ang)], 0) / N).float())   # (2F, N)
        s.register_buffer('win', w.float())
        Lfull = (frames - 1) * hop + N; ola = torch.zeros(frames * N, Lfull)
        for t in range(frames): ola[t * N:(t + 1) * N, t * hop:t * hop + N] = torch.eye(N)
        s.register_buffer('ola', ola)
        wss = (w.float() ** 2)[None, :].expand(frames, N).reshape(-1) @ ola
        s.register_buffer('wss', torch.where(wss > 1e-11, wss, torch.ones_like(wss)))
    def forward(s, z):   # z (..., F, T, 2) -> (..., length)
        lead = z.shape[:-3]; F_, T = z.shape[-3], z.shape[-2]
        zz = z.permute(*range(len(lead)), -2, -1, -3).reshape(-1, T, 2 * F_)
        fr = (zz @ s.idft) * s.win
        x = (fr.reshape(-1, T * s.N) @ s.ola) / s.wss
        x = x[:, s.N // 2:s.N // 2 + s.length]
        if x.shape[-1] < s.length: x = NF.pad(x, (0, s.length - x.shape[-1]))
        return x.reshape(*lead, s.length)

def cumsum_mat(n): return torch.triu(torch.ones(n, n))   # x @ M = cumsum along the last dim
class PhaseVocoder(nn.Module):
    """torchaudio.functional.phase_vocoder on (..., F, T, 2) for a fixed rate and frame count"""
    def __init__(s, rate, n_freq, hop, frames):
        super().__init__()
        ts = torch.arange(0, frames, rate, dtype=torch.float64)
        s.register_buffer('alphas', (ts % 1.0).float()); s.register_buffer('i0', ts.long()); s.register_buffer('i1', (ts + 1).long())
        s.register_buffer('adv', torch.linspace(0, math.pi * hop, n_freq)[:, None])
        s.register_buffer('cum', cumsum_mat(len(ts)))
    def forward(s, z):
        ph0 = cangle(z[..., :1, :]); zp = NF.pad(z, [0, 0, 0, 2])
        z0 = zp.index_select(-2, s.i0); z1 = zp.index_select(-2, s.i1)
        a0, a1, n0, n1 = cangle(z0), cangle(z1), cabs(z0), cabs(z1)
        ph = a1 - a0 - s.adv; ph = ph - 2 * math.pi * torch.round(ph / (2 * math.pi)); ph = ph + s.adv
        ph = torch.cat([ph0, ph[..., :-1]], -1); acc = ph @ s.cum
        mag = s.alphas * n1 + (1 - s.alphas) * n0
        return polar(mag, acc)

def iir_matrix(filt, length):
    """y = x @ M for a causal LTI filter `filt` (a torchaudio filtering function on (1, L) waveforms):
    M[i, j] = h[j - i], h the impulse response, so the filter is exact up to the signal length"""
    h = filt(torch.cat([torch.ones(1, 1, dtype=torch.float64), torch.zeros(1, length - 1, dtype=torch.float64)], 1))[0]
    M = torch.zeros(length, length, dtype=torch.float64)
    for i in range(length): M[i, i:] = h[:length - i]
    return M.float()

def psd_re(spec, mask, eps=1e-15):
    """torchaudio.functional.psd (normalize=True): spec (C, F, T, 2), mask (F, T) -> (F, C, C, 2)"""
    m = mask / (mask.sum(-1, keepdim=True) + eps)
    xr, xi = spec[..., 0], spec[..., 1]                                          # (C, F, T)
    xr = xr.permute(1, 0, 2); xi = xi.permute(1, 0, 2)                            # (F, C, T)
    mr = xr * m[:, None, :]; mi = xi * m[:, None, :]
    re = mr @ xr.transpose(-1, -2) + mi @ xi.transpose(-1, -2)                    # sum_t m x conj(x')
    im = mi @ xr.transpose(-1, -2) - mr @ xi.transpose(-1, -2)
    return torch.stack([re, im], -1)
def cinv2(M):
    """inverse of (..., 2, 2, 2) complex 2x2 matrices"""
    a, b, c, d = M[..., 0, 0, :], M[..., 0, 1, :], M[..., 1, 0, :], M[..., 1, 1, :]
    det = cmul(a, d) - cmul(b, c)
    adj = torch.stack([torch.stack([d, -b], -2), torch.stack([-c, a], -2)], -3)   # (..., 2, 2, 2)
    return cdiv(adj, det[..., None, None, :].expand_as(adj))
def cmm2(A, B):
    """(..., 2, 2, 2) @ (..., 2, k, 2) complex"""
    return torch.stack([sum(cmul(A[..., i, l, :], B[..., l, j, :]) for l in range(2)) for i in range(2) for j in range(B.shape[-2])], -2).reshape(*A.shape[:-3], 2, B.shape[-2], 2)
def tik(psd, reg=1e-7, eps=1e-8):
    tr = psd[..., 0, 0, 0] + psd[..., 1, 1, 0]; e = tr * reg + eps
    return psd + torch.stack([e[..., None, None] * torch.eye(2), torch.zeros_like(psd[..., 0])], -1)
def souden_weights(psd_s, psd_n, ref, eps=1e-8):
    num = cmm2(cinv2(tik(psd_n)), psd_s)                                          # (F, 2, 2, 2)
    tr = num[..., 0, 0, :] + num[..., 1, 1, :]
    ws = cdiv(num, (tr + torch.tensor([eps, 0.0]))[..., None, None, :].expand_as(num))
    return ws[..., :, ref, :]                                                      # (F, C, 2)
def rtf_weights(rtf, psd_n, ref, eps=1e-8):
    num = cmm2(cinv2(tik(psd_n)), rtf[..., :, None, :])[..., :, 0, :]             # (F, C, 2)
    den = cmul(cconj(rtf), num).sum(-2)                                            # (F, 2)
    w = num / (den[..., 0] + eps)[..., None, None]
    return cmul(w, cconj(rtf[..., ref, :])[..., None, :].expand_as(w))
def beamform(w, spec):
    """(F, C, 2) weights, (C, F, T, 2) spectrogram -> (F, T, 2): sum_c conj(w) X"""
    wc = cconj(w).permute(1, 0, 2)[..., None, :]                                  # (C, F, 1, 2)
    return cmul(wc.expand_as(spec), spec).sum(0)

def rnnt_loss(logits, targets, blank):
    """the transducer loss of one utterance on a (T, U + 1, C) lattice by the alpha recursion (fused log-softmax)"""
    lp = torch.log_softmax(logits, -1); T, U1, _ = lp.shape; U = U1 - 1
    alpha = [[None] * U1 for _ in range(T)]
    for t in range(T):
        for u in range(U1):
            terms = []
            if t == 0 and u == 0: alpha[t][u] = lp.new_zeros(()); continue
            if t > 0: terms.append(alpha[t - 1][u] + lp[t - 1, u, blank])
            if u > 0: terms.append(alpha[t][u - 1] + lp[t, u - 1, targets[u - 1]])
            alpha[t][u] = terms[0] if len(terms) == 1 else torch.logaddexp(terms[0], terms[1])
    return -(alpha[T - 1][U] + lp[T - 1, U, blank])
