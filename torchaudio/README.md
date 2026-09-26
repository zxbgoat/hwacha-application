# torchaudio models on Hwacha

The model classes of docs.pytorch.org/audio/stable/models.html (torchaudio 2.9), one case per class
(two for the wav2vec2 family's variants and Wav2Letter's two input types), run through PyTorch ->
torch-mlir -> hwacha-mlir -> hwacha-cc and checked on Spike against PyTorch (fixed seed). Layout,
generic host, `gen.sh` and Makefile are those of `../ttmodule`; the cases live in `export_ta.py`.

The page's factory functions (`wav2vec2_base`, `hdemucs_low`, `emformer_rnnt_base`, `conv_tasnet_base`,
`squim_objective_base` ...) build the paper-size models, far beyond Spike, so every case instantiates
the class, or its configurable builder (`wav2vec2_model`, `wavlm_model`, `hubert_pretrain_model`,
`emformer_rnnt_model`, `squim_objective_model`), at a tiny size: feature dim 16, 2 layers, 2 heads,
sequences of 8 to 20 frames, waveforms of 64 to 400 samples. Wav2Letter has a fixed architecture
(11 convolutions of 250 / 2000 channels, 23M parameters): its two cases run at full size on short
inputs, and their assembly (1.4 GB each) is not tracked; `make gen-wav2letter` regenerates it in
about four minutes. Every case is a single-input module `net(x)` returning one float tensor: lengths,
labels, target symbols, spectrograms and reference waveforms are constant buffers, multiple outputs
are flattened and concatenated. Models run in eval mode.

## Cases: 16, all PASS

| case | class (builder) | Hwacha |
|---|---|---|
| conformer | `Conformer` | PASS, max\|diff\| 0, 26,133 周期 |
| emformer | `Emformer` | PASS, max\|diff\| 0, 21,587 周期 |
| conv_tasnet | `ConvTasNet` | PASS, max\|diff\| 0, 52,080 周期 |
| deepspeech | `DeepSpeech` | PASS, max\|diff\| 0, 13,430 周期 |
| wav2letter | `Wav2Letter (mfcc)` | PASS, max\|diff\| 0, 26,664,613 周期 |
| wav2letter_waveform | `Wav2Letter (waveform)` | PASS, max\|diff\| 0, 3,404,042 周期 |
| hdemucs | `HDemucs` | PASS, max\|diff\| 0, 118,908 周期 |
| wav2vec2 | `Wav2Vec2Model (wav2vec2_model)` | PASS, max\|diff\| 1e-06, 28,202 周期 |
| wav2vec2_aux | `Wav2Vec2Model (aux CTC head)` | PASS, max\|diff\| 0, 28,574 周期 |
| wavlm | `Wav2Vec2Model (wavlm_model)` | PASS, max\|diff\| 1e-06, 31,058 周期 |
| hubert_pretrain | `HuBERTPretrainModel (hubert_pretrain_model)` | PASS, max\|diff\| 0, 33,568 周期 |
| rnnt | `RNNT (emformer_rnnt_model)` | PASS, max\|diff\| 0, 37,310 周期 |
| squim_objective | `SquimObjective (squim_objective_model)` | PASS, max\|diff\| 0, 306,128 周期 |
| squim_subjective | `SquimSubjective` | PASS, max\|diff\| 0, 52,419 周期 |
| tacotron2 | `Tacotron2` | PASS, max\|diff\| 0, 47,271 周期 |
| wavernn | `WaveRNN` | PASS, max\|diff\| 0, 36,283 周期 |

Not cases: `RNNTBeamSearch` (a Python beam search over hypothesis lists, data dependent; the RNNT
it drives is the `rnnt` case) and the streaming / autoregressive `infer` methods of Emformer,
Tacotron2 and WaveRNN (state lists, early stopping); the pretrained-weight pipelines
(`torchaudio.pipelines`) are not on this page.

**Exports that differ from the reference** (the reference is the genuine torchaudio forward,
asserted equal before export): `conformer` calls the layers without the padding mask that `forward`
derives from `lengths` (full lengths: no mask); `emformer` / `rnnt` export with torch-mlir's
decomposition table minus the tensor constructors (their decomposition of `aten.zeros` / `ones` yields
`aten.empty_strided`, which has no lowering; torch-mlir lowers the constructors directly); `hdemucs`
uses a subclass whose STFT / iSTFT are matmuls with the windowed DFT matrices, constant frame gathers,
an overlap-add matmul and torch.istft's window-sum-square normalisation, the spectrogram a real tensor
with a trailing (re, im) axis (torch.stft, hann_window and complex tensors have no lowering; within
2e-3 of the genuine model); `hubert_pretrain` pins the mask generator's random mask to the draw under
`torch.manual_seed(0)` (the reference is called under the same seed), applies the mask embedding with
`torch.where`, computes the logits for all frames and picks the masked / unmasked rows with constant
one-hot matmuls (the genuine code uses boolean indexing), with -1e4 in place of -inf and softmax over
the classes as the compared output; `tacotron2` runs the encoder without `pack_padded_sequence`, the
decoder with a constant all-false memory mask, and the prenet's always-on dropout disabled for both
the export and the reference (the genuine forward is otherwise not reproducible).
