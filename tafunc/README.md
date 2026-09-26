# torchaudio.transforms on Hwacha

The transforms of docs.pytorch.org/audio/stable/transforms.html (torchaudio 2.9), one case per class
(two for Spectrogram's power / complex outputs and Resample's two kernels), run through PyTorch ->
torch-mlir -> hwacha-mlir -> hwacha-cc and checked on Spike against PyTorch (fixed seed). Layout,
generic host, `gen.sh` and Makefile are those of `../torchaudio`; the cases live in `export_taf.py`,
the tensor re-implementations in `taf_lib.py`. Signals are small: sample rate 800, n_fft 32, hop 8,
128-sample waveforms (17 x 17 spectrograms), 8 mel bands. Second inputs (noise, kernels, masks, PSD
matrices, targets) are constant buffers, complex spectrograms travel as real tensors with a trailing
(re, im) axis, integer outputs are cast to float; the random transforms are pinned to one draw under
`torch.manual_seed(0)` with the reference called under the same seed.

## Cases: 37 (35 of the page's 36 classes), all PASS

| group | case | class | Hwacha |
|---|---|---|---|
| spectral | spectrogram | `Spectrogram` | PASS, max\|diff\| 5e-06, 1,673 周期 |
| spectral | spectrogram_complex | `Spectrogram (power=None)` | PASS, max\|diff\| 0, 1,444 周期 |
| spectral | inverse_spectrogram | `InverseSpectrogram` | PASS, max\|diff\| 0, 4,947 周期 |
| spectral | griffin_lim | `GriffinLim` | PASS, max\|diff\| 2e-06, 31,837 周期 |
| spectral | mel_scale | `MelScale` | PASS, max\|diff\| 1e-06, 437 周期 |
| spectral | inverse_mel_scale | `InverseMelScale` | PASS, max\|diff\| 1e-06, 281 周期 |
| spectral | mel_spectrogram | `MelSpectrogram` | PASS, max\|diff\| 7e-06, 2,091 周期 |
| spectral | mfcc | `MFCC` | PASS, max\|diff\| 3e-06, 3,140 周期 |
| spectral | lfcc | `LFCC` | PASS, max\|diff\| 3e-06, 3,140 周期 |
| spectral | spectral_centroid | `SpectralCentroid` | PASS, max\|diff\| 4.5e-05, 3,512 周期 |
| spectral | amplitude_to_db | `AmplitudeToDB` | PASS, max\|diff\| 1e-06, 1,033 周期 |
| spectral | compute_deltas | `ComputeDeltas` | PASS, max\|diff\| 0, 646 周期 |
| spectral | sliding_window_cmn | `SlidingWindowCmn` | PASS, max\|diff\| 5e-06, 1,307 周期 |
| spectral | time_stretch | `TimeStretch` | PASS, max\|diff\| 0.0001, 3,887 周期 |
| spectral | pitch_shift | `PitchShift` | PASS, max\|diff\| 7e-06, 406,161 周期 |
| masking | frequency_masking | `FrequencyMasking` | PASS, max\|diff\| 0, 78 周期 |
| masking | time_masking | `TimeMasking` | PASS, max\|diff\| 0, 78 周期 |
| masking | spec_augment | `SpecAugment` | PASS, max\|diff\| 0, 78 周期 |
| waveform | add_noise | `AddNoise` | PASS, max\|diff\| 0, 20,352 周期 |
| waveform | convolve | `Convolve` | PASS, max\|diff\| 0, 418 周期 |
| waveform | fft_convolve | `FFTConvolve` | PASS, max\|diff\| 0, 418 周期 |
| waveform | deemphasis | `Deemphasis` | PASS, max\|diff\| 2e-06, 1,070 周期 |
| waveform | preemphasis | `Preemphasis` | PASS, max\|diff\| 0, 223 周期 |
| waveform | fade | `Fade` | PASS, max\|diff\| 0, 1,005 周期 |
| waveform | vol | `Vol` | PASS, max\|diff\| 0, 125 周期 |
| waveform | mu_law_encoding | `MuLawEncoding` | PASS, max\|diff\| 0, 807 周期 |
| waveform | mu_law_decoding | `MuLawDecoding` | PASS, max\|diff\| 0, 812 周期 |
| waveform | resample | `Resample` | PASS, max\|diff\| 0, 493 周期 |
| waveform | resample_kaiser | `Resample (kaiser)` | PASS, max\|diff\| 0, 461 周期 |
| waveform | speed | `Speed` | PASS, max\|diff\| 0, 481 周期 |
| waveform | speed_perturbation | `SpeedPerturbation` | PASS, max\|diff\| 0, 517 周期 |
| waveform | loudness | `Loudness` | PASS, max\|diff\| 6e-05, 8,127 周期 |
| multichannel | psd | `PSD` | PASS, max\|diff\| 5e-06, 2,181 周期 |
| multichannel | mvdr | `MVDR` | PASS, max\|diff\| 0, 13,397 周期 |
| multichannel | souden_mvdr | `SoudenMVDR` | PASS, max\|diff\| 1e-06, 9,393 周期 |
| multichannel | rtf_mvdr | `RTFMVDR` | PASS, max\|diff\| 1e-06, 7,800 周期 |
| loss | rnnt_loss | `RNNTLoss` | PASS, max\|diff\| 0, 1,949 周期 |

Not a case: `Vad` (a trigger search with data-dependent loops and no fixed output length).

**Exports that differ from the reference** (the reference is always the genuine transform, asserted
equal before export, within 1e-3 to 2e-3 for the spectral and multichannel cases):
- STFT: torch-mlir lowers `torch.stft` through `view_as_real`, but the result keeps complex types and
  dynamic shapes hwacha-mlir does not take, and complex `abs`, `istft`, `angle`, `conj` and
  `fft_irfft` have no lowering at all. `taf_lib.STFT` gathers the frames with a constant index and
  multiplies by the windowed DFT matrix (center, reflect padding, hann window); `taf_lib.ISTFT` does
  the c_f-weighted inverse DFT matmul, the window, an overlap-add matmul and torch.istft's
  window-sum-square normalisation. Spectrogram / MelSpectrogram / MFCC / LFCC / SpectralCentroid /
  InverseSpectrogram / GriffinLim (4 unrolled iterations, momentum 0.99 / 1.99) / PitchShift are built on them;
  TimeStretch and PitchShift use a real-valued phase vocoder (atan with quadrant fix-ups, a
  triangular-matmul cumsum, cos / sin for polar).
- `InverseMelScale`'s lstsq (driver gels, the minimum-norm solution) is a constant pseudo-inverse;
  `FFTConvolve` is the direct convolution (equal in full mode); `Deemphasis` and the two K-weighting
  biquads of `Loudness` are impulse-response matmuls clamped to [-1, 1] like lfilter (the Python
  lfilter breaks under torch.export), Loudness' 400 ms blocks a constant framing index and its two
  gating stages masked means; `SlidingWindowCmn`'s sliding sums are a constant averaging matrix (the
  loop's cumsum exports as tm_tensor.scan); the masking transforms multiply by the 0/1 mask of the
  seeded draw, `SpeedPerturbation` runs the speeder the seeded draw selects.
- `PSD` / `MVDR` / `SoudenMVDR` / `RTFMVDR` are written in real arithmetic on (re, im) pairs, the
  complex 2x2 solves as closed-form inverses with the diagonal loading, trace normalisation and
  conj(w)·X of torchaudio.functional (MVDR computes in double, the export in float); `RNNTLoss` (a
  C++ loss) is the transducer alpha recursion unrolled over the 4 x 3 lattice with fused log-softmax.
