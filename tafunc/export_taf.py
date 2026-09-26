#!/usr/bin/env python3
"""Export one torchaudio.transforms transform (docs.pytorch.org/audio/stable/transforms.html) through
torch-mlir to linalg-on-tensors, plus a PyTorch reference for one call. Small signals (sample rate
800, n_fft 32, hop 8, 128-sample waveforms, 8 mel bands); second inputs (noise, kernels, masks, PSD
matrices, targets) are constant buffers, complex spectrograms travel as real tensors with a trailing
(re, im) axis, integer outputs are cast to float. Random transforms are pinned to one draw under
torch.manual_seed(0) (the reference is called under the same seed).
usage: export_taf.py <name> <mlir_out> <check_out>"""
import sys, os, math, struct, numpy as np, torch, torch.nn as nn, torch.nn.functional as NF
import torchaudio.transforms as T, torchaudio.functional as AF
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import taf_lib as LB
from torch_mlir import fx

class Case(nn.Module):
    def __init__(s, body, mods=None, **bufs):
        super().__init__(); s.body = body
        for k, v in (mods() if callable(mods) else (mods or {})).items(): setattr(s, k, v)
        for k, v in bufs.items(): s.register_buffer(k, v)
        for m in s.modules():
            for n in list(m._non_persistent_buffers_set): t = getattr(m, n); delattr(m, n); m.register_buffer(n, t)
    def forward(s, x): return s.body(s, x)
class Ref(Case):
    def __init__(s, body, ref, mods=None, _atol=1e-4, **bufs): super().__init__(body, mods, **bufs); s.ref = ref; s.atol = _atol
    def reference(s, x): return s.ref(s, x)
def export_linalg(m, x):
    from torch_mlir.extras.fx_decomp_util import get_decomposition_table
    table = {k: v for k, v in get_decomposition_table().items() if not any(str(k).startswith('aten.' + n) for n in ('zeros', 'ones', 'full', 'new_zeros', 'new_ones', 'new_full', 'empty_like', 'zeros_like', 'ones_like', 'full_like'))}
    return fx.export_and_import(torch.export.export(m, (x,)), x, output_type='linalg-on-tensors', func_name='net', decomposition_table=table)
def R(*shape): return torch.randn(*shape)
def seeded(f):
    def g(s, x): torch.manual_seed(0); return f(s, x)
    return g

SR, NFFT, HOP, L, NMEL = 800, 32, 8, 128, 8
NFREQ = NFFT // 2 + 1

def build(name):
    torch.manual_seed(0)
    C = {}
    def case(n, body, x, mods=None, **b): C[n] = lambda: (Case(body, mods, **b), x)
    def rcase(n, body, ref, x, mods=None, **b): C[n] = lambda: (Ref(body, ref, mods, **b), x)
    wav = R(1, L) * 0.5
    spec = T.Spectrogram(n_fft=NFFT, hop_length=HOP)(wav)                              # (1, 17, 17) power spectrogram
    cspec = T.Spectrogram(n_fft=NFFT, hop_length=HOP, power=None)(wav)                 # complex
    cre = torch.view_as_real(cspec)                                                     # (1, 17, 17, 2)
    NT = spec.shape[-1]
    def cplx(x): return torch.view_as_complex(x.contiguous())
    win = LB.hann(NFFT); stft = lambda: LB.STFT(NFFT, HOP, L)
    def pw(s, x): return LB.power(s.stft(x))
    def melf(s, x): return (pw(s, x).transpose(-1, -2) @ s.m.mel_scale.fb).transpose(-1, -2)
    # ---- spectrograms and scales (torch.stft exports through view_as_real; complex abs, istft, angle do not)
    rcase('spectrogram', pw, lambda s, x: s.m(x), wav, lambda: dict(m=T.Spectrogram(n_fft=NFFT, hop_length=HOP), stft=stft()))
    rcase('spectrogram_complex', lambda s, x: s.stft(x), lambda s, x: torch.view_as_real(s.m(x)), wav, lambda: dict(m=T.Spectrogram(n_fft=NFFT, hop_length=HOP, power=None), stft=stft()))
    rcase('inverse_spectrogram', lambda s, x: s.inv(x), lambda s, x: s.m(cplx(x), length=L), cre, lambda: dict(m=T.InverseSpectrogram(n_fft=NFFT, hop_length=HOP), inv=LB.ISTFT(NFFT, HOP, NT, L)), _atol=1e-3)
    def gl_body(s, x):   # torchaudio.functional.griffinlim, n_iter 4, momentum 0.99 / (1 + 0.99), angles initialised to 1
        mag = x.pow(0.5); ang = torch.stack([torch.ones_like(mag), torch.zeros_like(mag)], -1); tprev = torch.zeros_like(ang)
        for _ in range(4):
            inv = s.inv(LB.cscale(ang, mag)); reb = s.stft(inv)
            ang = reb - (0.99 / 1.99) * tprev; ang = ang / (LB.cabs(ang) + 1e-16)[..., None]; tprev = reb
        return s.inv(LB.cscale(ang, mag))
    rcase('griffin_lim', gl_body, lambda s, x: s.m(x), spec, lambda: dict(m=T.GriffinLim(n_fft=NFFT, hop_length=HOP, n_iter=4, rand_init=False, length=L), inv=LB.ISTFT(NFFT, HOP, NT, L), stft=stft()), _atol=1e-3)
    case('mel_scale', lambda s, x: s.m(x), spec, lambda: dict(m=T.MelScale(n_mels=NMEL, sample_rate=SR, n_stft=NFREQ)))
    mel = T.MelScale(n_mels=NMEL, sample_rate=SR, n_stft=NFREQ)(spec)
    def ims_mods():   # lstsq (driver gels: the minimum-norm solution of the underdetermined system) as a constant pseudo-inverse
        m = T.InverseMelScale(n_stft=NFREQ, n_mels=NMEL, sample_rate=SR); A = m.fb.double().t(); P = (A.t() @ torch.linalg.inv(A @ A.t())).float()
        assert torch.allclose(torch.relu(P @ mel), m(mel), atol=1e-4), 'pseudo-inverse != lstsq'; return dict(m=m, P=P)
    rcase('inverse_mel_scale', lambda s, x: torch.relu(s.P @ x), lambda s, x: s.m(x), mel, ims_mods)
    rcase('mel_spectrogram', melf, lambda s, x: s.m(x), wav, lambda: dict(m=T.MelSpectrogram(sample_rate=SR, n_fft=NFFT, hop_length=HOP, n_mels=NMEL), stft=stft()))
    def mfcc_body(s, x):
        m = s.m; ml = (pw(s, x).transpose(-1, -2) @ m.MelSpectrogram.mel_scale.fb).transpose(-1, -2)
        return (m.amplitude_to_DB(ml).transpose(-1, -2) @ m.dct_mat).transpose(-1, -2)
    rcase('mfcc', mfcc_body, lambda s, x: s.m(x), wav, lambda: dict(m=T.MFCC(sample_rate=SR, n_mfcc=4, melkwargs=dict(n_fft=NFFT, hop_length=HOP, n_mels=NMEL)), stft=stft()), _atol=1e-3)
    def lfcc_body(s, x):
        m = s.m; lf = (pw(s, x).transpose(-1, -2) @ m.filter_mat).transpose(-1, -2)
        return (m.amplitude_to_DB(lf).transpose(-1, -2) @ m.dct_mat).transpose(-1, -2)
    rcase('lfcc', lfcc_body, lambda s, x: s.m(x), wav, lambda: dict(m=T.LFCC(sample_rate=SR, n_filter=NMEL, n_lfcc=4, speckwargs=dict(n_fft=NFFT, hop_length=HOP)), stft=stft()), _atol=1e-3)
    rcase('spectral_centroid', lambda s, x: (lambda mg: (s.freqs[:, None] * mg).sum(-2) / mg.sum(-2))(LB.cabs(s.stft(x))), lambda s, x: s.m(x), wav,
          lambda: dict(m=T.SpectralCentroid(sample_rate=SR, n_fft=NFFT, hop_length=HOP), stft=stft()), freqs=torch.linspace(0, SR // 2, NFREQ), _atol=1e-3)
    case('amplitude_to_db', lambda s, x: s.m(x), spec, lambda: dict(m=T.AmplitudeToDB('power', top_db=80.0)))
    case('compute_deltas', lambda s, x: s.m(x), spec, lambda: dict(m=T.ComputeDeltas(win_length=5)))
    def cmn_windows(T, cmn_window, min_cmn_window, center):   # the per-frame [start, end) of torchaudio.functional.sliding_window_cmn
        W = torch.zeros(T, T)
        for t in range(T):
            ws, we = (t - cmn_window // 2, t - cmn_window // 2 + cmn_window) if center else (t - cmn_window, t + 1)
            if ws < 0: we -= ws; ws = 0
            if not center and we > t: we = max(t + 1, min_cmn_window)
            if we > T: ws -= we - T; we = T; ws = max(ws, 0)
            W[t, ws:we] = 1.0 / (we - ws)
        return W
    def cmn_body(s, x):   # the sliding sums as a constant averaging matmul (the genuine loop's cumsum exports as tm_tensor.scan)
        mean = s.W @ x; var = s.W @ (x * x) - mean * mean; return (x - mean) * var.pow(-0.5)
    rcase('sliding_window_cmn', cmn_body, lambda s, x: s.m(x), spec.transpose(1, 2), lambda: dict(m=T.SlidingWindowCmn(cmn_window=6, min_cmn_window=2, center=True, norm_vars=True)), W=cmn_windows(NT, 6, 2, True), _atol=1e-3)
    rcase('time_stretch', lambda s, x: s.pv(x), lambda s, x: torch.view_as_real(s.m(cplx(x))), cre, lambda: dict(m=T.TimeStretch(hop_length=HOP, n_freq=NFREQ, fixed_rate=1.25), pv=LB.PhaseVocoder(1.25, NFREQ, HOP, NT)), _atol=1e-3)
    # ---- masking: the mask of the draw under seed 0 (from the transform applied to ones), applied by multiplication
    def masked(n, mk):
        def mods():
            m = mk(); torch.manual_seed(0); return dict(m=m, mask=m(torch.ones_like(spec)))
        rcase(n, lambda s, x: x * s.mask, seeded(lambda s, x: s.m(x)), spec, mods)
    masked('frequency_masking', lambda: T.FrequencyMasking(freq_mask_param=6))
    masked('time_masking', lambda: T.TimeMasking(time_mask_param=6))
    masked('spec_augment', lambda: T.SpecAugment(n_time_masks=2, time_mask_param=4, n_freq_masks=2, freq_mask_param=4, zero_masking=True))
    # ---- waveform transforms
    case('add_noise', lambda s, x: s.m(x, s.noise, s.snr), wav, lambda: dict(m=T.AddNoise()), noise=R(1, L) * 0.1, snr=torch.tensor([10.0]))
    case('convolve', lambda s, x: s.m(x, s.k), wav, lambda: dict(m=T.Convolve('full')), k=R(1, 9) * 0.3)
    rcase('fft_convolve', lambda s, x: AF.convolve(x, s.k, 'full'), lambda s, x: s.m(x, s.k), wav, lambda: dict(m=T.FFTConvolve('full')), k=R(1, 9) * 0.3)
    rcase('deemphasis', lambda s, x: (x @ s.M).clamp(-1, 1), lambda s, x: s.m(x), wav, lambda: dict(m=T.Deemphasis(0.97), M=LB.iir_matrix(lambda w: AF.deemphasis(w, 0.97), L)))   # lfilter clamps its output to [-1, 1]
    case('preemphasis', lambda s, x: s.m(x), wav, lambda: dict(m=T.Preemphasis(0.97)))
    case('fade', lambda s, x: s.m(x), wav, lambda: dict(m=T.Fade(fade_in_len=32, fade_out_len=32, fade_shape='linear')))
    case('vol', lambda s, x: s.m(x), wav, lambda: dict(m=T.Vol(gain=2.0, gain_type='amplitude')))
    case('mu_law_encoding', lambda s, x: s.m(x).float(), wav, lambda: dict(m=T.MuLawEncoding(256)))
    case('mu_law_decoding', lambda s, x: s.m(x.long()), T.MuLawEncoding(256)(wav).float(), lambda: dict(m=T.MuLawDecoding(256)))
    case('resample', lambda s, x: s.m(x), wav, lambda: dict(m=T.Resample(orig_freq=8, new_freq=6)))
    case('resample_kaiser', lambda s, x: s.m(x), wav, lambda: dict(m=T.Resample(orig_freq=4, new_freq=5, resampling_method='sinc_interp_kaiser')))
    case('speed', lambda s, x: s.m(x)[0], wav, lambda: dict(m=T.Speed(orig_freq=SR, factor=1.25)))
    def sp_mods():
        m = T.SpeedPerturbation(orig_freq=SR, factors=[0.9, 1.0, 1.1]); torch.manual_seed(0); idx = int(torch.randint(3, ())); return dict(m=m, sp=m.speeders[idx])
    rcase('speed_perturbation', lambda s, x: s.sp(x)[0], seeded(lambda s, x: s.m(x)[0]), wav, sp_mods)
    rate = 2.0 ** (-2 / 12); len_st = int(round(L / rate))
    def ps_body(s, x):   # torchaudio.functional.pitch_shift: stft -> phase vocoder (rate 2^(-2/12)) -> istft -> resample -> crop to the input length
        st = s.inv(s.pv(s.stft(x))); sh = AF.resample(st, int(SR / rate), SR); return sh[..., :L]
    rcase('pitch_shift', ps_body, lambda s, x: s.m(x), wav, lambda: dict(m=T.PitchShift(sample_rate=SR, n_steps=2, n_fft=NFFT, hop_length=HOP), pv=LB.PhaseVocoder(rate, NFREQ, HOP, NT), inv=LB.ISTFT(NFFT, HOP, len(torch.arange(0, NT, rate)), len_st), stft=stft()), _atol=2e-3)
    # loudness (ITU-R BS.1770): the K-weighting biquads as one impulse-response matmul, the 400 ms blocks
    # by a constant framing index, the two gating stages as masked means
    LS, LR = 400, 200; gate = int(round(0.4 * LR)); step = int(round(gate * 0.25)); nb = (LS - gate) // step + 1
    def loud_body(s, x):
        y = ((x @ s.K1).clamp(-1, 1) @ s.K2).clamp(-1, 1); e = (y[0].index_select(0, s.fidx).reshape(nb, gate) ** 2).mean(-1)   # each biquad's lfilter clamps to [-1, 1]
        loud = -0.691 + 10 * torch.log10(e)
        g1 = (loud > -70).float(); e1 = (g1 * e).sum() / g1.sum(); rel = -0.691 + 10 * torch.log10(e1) - 10
        g2 = g1 * (loud > rel).float(); e2 = (g2 * e).sum() / g2.sum()
        return (-0.691 + 10 * torch.log10(e2)).reshape(1)
    rcase('loudness', loud_body, lambda s, x: s.m(x).reshape(1), R(1, LS) * 0.3, lambda: dict(m=T.Loudness(sample_rate=LR), K1=LB.iir_matrix(lambda w: AF.treble_biquad(w, LR, 4.0, 1500.0, 1 / math.sqrt(2)), LS), K2=LB.iir_matrix(lambda w: AF.highpass_biquad(w, LR, 38.0, 0.5), LS)),
          fidx=(torch.arange(nb)[:, None] * step + torch.arange(gate)[None, :]).reshape(-1), _atol=1e-3)
    # ---- multichannel (2 channels, complex spectrograms as (re, im)); MVDR computes in double, the export in float
    mc = torch.view_as_real(T.Spectrogram(n_fft=NFFT, hop_length=HOP, power=None)(R(2, L)))   # (2, 17, 17, 2)
    mask = torch.rand(NFREQ, NT)
    rcase('psd', lambda s, x: LB.psd_re(x, s.mask), lambda s, x: torch.view_as_real(s.m(cplx(x), s.mask)), mc, lambda: dict(m=T.PSD()), mask=mask, _atol=1e-3)
    rcase('mvdr', lambda s, x: LB.beamform(LB.souden_weights(LB.psd_re(x, s.mask), LB.psd_re(x, 1 - s.mask), 0), x), lambda s, x: torch.view_as_real(s.m(cplx(x), s.mask, 1 - s.mask)), mc,
          lambda: dict(m=T.MVDR(ref_channel=0, solution='ref_channel')), mask=mask, _atol=2e-3)
    psd_s = T.PSD()(cplx(mc), mask); psd_n = T.PSD()(cplx(mc), 1 - mask)
    rcase('souden_mvdr', lambda s, x: LB.beamform(LB.souden_weights(s.ps, s.pn, 0), x), lambda s, x: torch.view_as_real(s.m(cplx(x), cplx(s.ps), cplx(s.pn), 0)), mc,
          lambda: dict(m=T.SoudenMVDR()), ps=torch.view_as_real(psd_s), pn=torch.view_as_real(psd_n), _atol=2e-3)
    rtf = AF.rtf_evd(psd_s)
    rcase('rtf_mvdr', lambda s, x: LB.beamform(LB.rtf_weights(s.rtf, s.pn, 0), x), lambda s, x: torch.view_as_real(s.m(cplx(x), cplx(s.rtf), cplx(s.pn), 0)), mc,
          lambda: dict(m=T.RTFMVDR()), rtf=torch.view_as_real(rtf), pn=torch.view_as_real(psd_n), _atol=2e-3)
    # ---- RNN-T loss on a tiny lattice (the C++ loss has no export path): the alpha recursion unrolled
    logits = R(1, 4, 3, 5); targets = torch.tensor([[1, 2]], dtype=torch.int32)
    rcase('rnnt_loss', lambda s, x: LB.rnnt_loss(x[0], [1, 2], 0).reshape(1), lambda s, x: s.m(x, s.t, s.ll, s.tl).reshape(1), logits, lambda: dict(m=T.RNNTLoss(blank=0)), t=targets, ll=torch.tensor([4], dtype=torch.int32), tl=torch.tensor([2], dtype=torch.int32))
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
    mod = export_linalg(m, x)
    open(mlir_out, 'w').write(str(mod))
    xf = np.ascontiguousarray(x.numpy()).astype(np.float32).ravel(); yf = np.ascontiguousarray(y.numpy()).astype(np.float32).ravel()
    with open(bin_out, 'wb') as f:
        f.write(struct.pack('i', xf.size)); f.write(xf.tobytes()); f.write(struct.pack('i', yf.size)); f.write(yf.tobytes())
