#!/usr/bin/env python3
"""Export one torchaudio model (docs.pytorch.org/audio/stable/models.html) through torch-mlir to
linalg-on-tensors, plus a PyTorch reference for one call. The page's classes are instantiated at tiny
sizes (their factory functions build the paper-size models: far beyond Spike), with the class' own
builder where one exists (wav2vec2_model, hubert_pretrain_model, wavlm_model, emformer_rnnt_model,
squim_objective_model). Every case is a single-input module net(x) -> one float tensor: lengths,
targets, spectrograms and token ids are constant buffers, multiple outputs are flattened and
concatenated. Models run in eval mode.       usage: export_ta.py <name> <mlir_out> <check_out>"""
import sys, os, struct, math, copy, numpy as np, torch, torch.nn as nn, torch.nn.functional as NF
from torch_mlir import fx
import torchaudio.models as A

class Case(nn.Module):
    def __init__(s, body, mods=None, **bufs):
        super().__init__(); s.body = body
        for k, v in (mods() if callable(mods) else (mods or {})).items(): setattr(s, k, v)
        for k, v in bufs.items(): s.register_buffer(k, v)
        for m in s.modules():   # non-persistent buffers must be in the state dict for torch-mlir's importer
            for n in list(m._non_persistent_buffers_set): t = getattr(m, n); delattr(m, n); m.register_buffer(n, t)
    def forward(s, x): return s.body(s, x)
class Ref(Case):
    """exported body differs from the reference (the genuine torchaudio forward, what check.bin holds)"""
    def __init__(s, body, ref, mods=None, _atol=1e-4, **bufs): super().__init__(body, mods, **bufs); s.ref = ref; s.atol = _atol
    def reference(s, x): return s.ref(s, x)
def export_linalg(m, x):
    """torch.export, then torch-mlir with its decomposition table minus the tensor constructors: their
    decompositions produce aten.empty_strided (the Emformer's zeros / ones masks), which has no lowering,
    while torch-mlir lowers aten.zeros / ones / full directly"""
    from torch_mlir.extras.fx_decomp_util import get_decomposition_table
    table = {k: v for k, v in get_decomposition_table().items() if not any(str(k).startswith('aten.' + n) for n in ('zeros', 'ones', 'full', 'new_zeros', 'new_ones', 'new_full', 'empty_like', 'zeros_like', 'ones_like', 'full_like'))}
    ep = torch.export.export(m, (x,))
    return fx.export_and_import(ep, x, output_type='linalg-on-tensors', func_name='net', decomposition_table=table)
def R(*shape): return torch.randn(*shape)
def flat(*ts): return torch.cat([t.reshape(-1) for t in ts])

class HDemucsExport(A.HDemucs):
    """HDemucs with the STFT / iSTFT as matmuls (torch.stft, torch.hann_window and complex tensors have no
    torch-mlir lowering): the spectrogram is a real tensor with a trailing (re, im) axis, the frames are
    gathered with a constant index, transformed with the windowed DFT matrix, and the inverse does the
    windowed inverse DFT, an overlap-add matmul and the window-sum-square normalisation of torch.istft."""
    def _spec(self, x):
        hl, nfft = self.hop_length, self.nfft
        le = int(math.ceil(x.shape[-1] / hl)); pad = hl // 2 * 3
        x = self._pad1d(x, pad, pad + le * hl - x.shape[-1], mode='reflect')
        B, C, L = x.shape
        xp = NF.pad(x.reshape(-1, L), (nfft // 2, nfft // 2), mode='reflect')
        fr = xp.index_select(1, self.frame_idx).reshape(B * C, -1, nfft)             # (BC, T, N)
        z = (fr @ self.dft).reshape(B, C, -1, 2, nfft // 2 + 1).permute(0, 1, 4, 2, 3)   # (B, C, F, T, 2)
        z = z[:, :, :-1]
        return z[:, :, :, 2:2 + le]
    def _magnitude(self, z):
        B, C, Fr, T, _ = z.shape
        return z.permute(0, 1, 4, 2, 3).reshape(B, C * 2, Fr, T)
    def _mask(self, m):
        B, S, C, Fr, T = m.shape
        return m.view(B, S, -1, 2, Fr, T).permute(0, 1, 2, 4, 5, 3)
    def _ispec(self, z, length=None):
        hl, nfft = self.hop_length, self.nfft
        z = NF.pad(z, [0, 0, 0, 0, 0, 1]); z = NF.pad(z, [0, 0, 2, 2])                # (B, S, C, F, T, 2)
        pad = hl // 2 * 3; le = hl * int(math.ceil(length / hl)) + 2 * pad
        B, S, C, F_, T, _ = z.shape
        zz = z.permute(0, 1, 2, 4, 5, 3).reshape(B * S * C, T, 2 * F_)
        fr = (zz @ self.idft) * self.win                                            # (M, T, N)
        x = (fr.reshape(B * S * C, T * nfft) @ self.ola) / self.wss                  # (M, Lfull)
        x = x[:, nfft // 2:nfft // 2 + le].reshape(B, S, C, le)
        return x[..., pad:pad + length]
def hdemucs_export(m, length):
    e = copy.deepcopy(m); e.__class__ = HDemucsExport
    N, hl = m.nfft, m.hop_length; F_ = N // 2 + 1
    le = int(math.ceil(length / hl)); pad = hl // 2 * 3; L = 2 * pad + le * hl + N; T = 1 + (L - N) // hl
    n = torch.arange(N, dtype=torch.float64); w = 0.5 - 0.5 * torch.cos(2 * math.pi * n / N)
    ang = 2 * math.pi * torch.outer(n, torch.arange(F_, dtype=torch.float64)) / N
    e.register_buffer('frame_idx', (torch.arange(T)[:, None] * hl + torch.arange(N)[None, :]).reshape(-1))
    e.register_buffer('dft', torch.cat([w[:, None] * torch.cos(ang), -w[:, None] * torch.sin(ang)], 1).float() / math.sqrt(N))
    c = torch.full((F_,), 2.0, dtype=torch.float64); c[0] = 1; c[-1] = 1
    e.register_buffer('idft', torch.cat([c[:, None] * torch.cos(ang.t()), -c[:, None] * torch.sin(ang.t())], 0).float() / math.sqrt(N))
    e.register_buffer('win', w.float())
    Ti = le + 4; Lfull = (Ti - 1) * hl + N
    ola = torch.zeros(Ti * N, Lfull)
    for t in range(Ti): ola[t * N:(t + 1) * N, t * hl:t * hl + N] = torch.eye(N)
    e.register_buffer('ola', ola)
    wss = (w.float() ** 2)[None, :].expand(Ti, N).reshape(-1) @ ola
    e.register_buffer('wss', torch.where(wss > 1e-11, wss, torch.ones_like(wss)))
    return e

def build(name):
    torch.manual_seed(0)
    C = {}
    def case(n, body, x, mods=None, **b): C[n] = lambda: (Case(body, mods, **b), x)
    def rcase(n, body, ref, x, mods=None, **b): C[n] = lambda: (Ref(body, ref, mods, **b), x)
    T, D = 12, 16
    # ---- Conformer / Emformer: (B, T, D) frames with lengths; the (output, lengths) pair -> output
    def conformer_body(s, x):   # Conformer.forward builds the padding mask from lengths (data dependent); full lengths: no mask
        h = x.transpose(0, 1)
        for layer in s.m.conformer_layers: h = layer(h, None)
        return h.transpose(0, 1)
    rcase('conformer', conformer_body, lambda s, x: s.m(x, s.len)[0], R(1, T, D), lambda: dict(m=A.Conformer(input_dim=D, num_heads=2, ffn_dim=32, num_layers=2, depthwise_conv_kernel_size=7)), len=torch.tensor([T]))
    case('emformer', lambda s, x: s.m(x, s.len)[0], R(1, T + 2, D), lambda: dict(m=A.Emformer(input_dim=D, num_heads=2, ffn_dim=32, num_layers=2, segment_length=4, right_context_length=2, left_context_length=4, max_memory_size=1)), len=torch.tensor([T]))
    # ---- ConvTasNet: (B, 1, samples) -> (B, sources, samples)
    case('conv_tasnet', lambda s, x: s.m(x), R(1, 1, 64), lambda: dict(m=A.ConvTasNet(num_sources=2, enc_kernel_size=4, enc_num_feats=16, msk_kernel_size=3, msk_num_feats=8, msk_num_hidden_feats=16, msk_num_layers=2, msk_num_stacks=1)))
    # ---- DeepSpeech / Wav2Letter
    case('deepspeech', lambda s, x: s.m(x), R(1, 1, 8, D), lambda: dict(m=A.DeepSpeech(n_feature=D, n_hidden=32, n_class=10)))
    case('wav2letter', lambda s, x: s.m(x), R(1, 13, 64), lambda: dict(m=A.Wav2Letter(num_classes=10, input_type='mfcc', num_features=13)))
    case('wav2letter_waveform', lambda s, x: s.m(x), R(1, 1, 1000), lambda: dict(m=A.Wav2Letter(num_classes=10, input_type='waveform', num_features=1)))
    # ---- HDemucs: (B, channels, samples) -> (B, sources, channels, samples)
    def hdemucs_mods():
        m = A.HDemucs(sources=['drums', 'vocals'], audio_channels=1, channels=4, nfft=64, depth=2)
        return dict(m=m, e=hdemucs_export(m, 256))
    rcase('hdemucs', lambda s, x: s.e(x), lambda s, x: s.m(x), R(1, 1, 256), hdemucs_mods, _atol=2e-3)
    # ---- wav2vec2 family: tiny feature extractor (3 convs) + 2-layer transformer, 400-sample waveform
    conv_cfg = [(8, 10, 5), (8, 3, 2), (8, 2, 2)]
    w2v = dict(extractor_mode='group_norm', extractor_conv_layer_config=conv_cfg, extractor_conv_bias=False, encoder_embed_dim=D, encoder_projection_dropout=0.0, encoder_pos_conv_kernel=8, encoder_pos_conv_groups=2,
               encoder_num_layers=2, encoder_num_heads=2, encoder_attention_dropout=0.0, encoder_ff_interm_features=32, encoder_ff_interm_dropout=0.0, encoder_dropout=0.0, encoder_layer_norm_first=False, encoder_layer_drop=0.0)
    wav = R(1, 400)
    case('wav2vec2', lambda s, x: s.m(x)[0], wav, lambda: dict(m=A.wav2vec2_model(aux_num_out=None, **w2v)))
    case('wav2vec2_aux', lambda s, x: s.m(x)[0], wav, lambda: dict(m=A.wav2vec2_model(aux_num_out=10, **w2v)))
    case('wavlm', lambda s, x: s.m(x)[0], wav, lambda: dict(m=A.wavlm_model(encoder_num_buckets=8, encoder_max_distance=16, aux_num_out=None, **w2v)))
    hub = dict(w2v, mask_prob=0.0, mask_selection='static', mask_other=0.0, mask_length=2, no_mask_overlap=False, mask_min_space=1, mask_channel_prob=0.0, mask_channel_selection='static', mask_channel_other=0.0,
               mask_channel_length=2, no_mask_channel_overlap=False, mask_channel_min_space=1, skip_masked=False, skip_nomask=False, num_classes=10, final_dim=8, feature_grad_mult=None)
    labels = torch.randint(0, 10, (1, 19))   # one label per extractor frame (400 samples -> 19)
    hub['mask_prob'] = 0.5
    from torchaudio.models.wav2vec2.components import _compute_mask_indices
    def hub_mods():
        m = A.hubert_pretrain_model(**hub); torch.manual_seed(0); mask = _compute_mask_indices((1, 19), None, 0.5, 2, 'static', 0.0, min_masks=2, no_overlap=False, min_space=1)
        T = mask.shape[1]; return dict(m=m, mask=mask, sel_m=torch.eye(T)[mask[0]], sel_u=torch.eye(T)[~mask[0]])   # the mask draw the reference will make under the same seed, and the row selections
    def hub_ref(s, x):
        torch.manual_seed(0); lm, lu, pen = s.m(x, s.lab); return flat(torch.softmax(lm, -1), torch.softmax(lu, -1), pen.reshape(1))
    def hub_body(s, x):   # HuBERTPretrainModel.forward with the mask a constant and the masked / unmasked rows picked by one-hot matmuls
        m = s.m; mask = s.mask
        f, _ = m.wav2vec2.feature_extractor(x, None); pen = f.pow(2).mean()
        h, _ = m.wav2vec2.encoder._preprocess(f, None)
        h = torch.where(mask[..., None], m.mask_generator.mask_embedding.to(h.dtype), h)
        h = m.wav2vec2.encoder.transformer(h, attention_mask=None)
        proj = m.logit_generator.final_proj(h)[0]; emb = m.logit_generator.label_embeddings; lab = s.lab[0]
        pos = emb.index_select(0, lab); negs = emb[:, None, :].expand(-1, proj.shape[0], -1)
        neg_is_pos = (pos[None] == negs).all(-1); targets = torch.cat([pos[None], negs], 0)
        logits = torch.cosine_similarity(proj[None], targets, dim=-1) / 0.1
        # the genuine code fills these with -inf; -1e4 gives the same softmax (exp underflows to 0) and survives the 0 * (-inf) of the selection matmul
        logits = torch.cat([logits[:1], torch.where(neg_is_pos, torch.full_like(logits[1:], -1e4), logits[1:])], 0).transpose(0, 1)
        return flat(torch.softmax(s.sel_m @ logits, -1), torch.softmax(s.sel_u @ logits, -1), pen.reshape(1))
    rcase('hubert_pretrain', hub_body, hub_ref, wav, hub_mods, lab=labels)
    # ---- RNN-T (Emformer transcriber + LSTM predictor + joiner): the joiner output (B, T', U + 1, symbols)
    rnnt_kw = dict(input_dim=D, encoding_dim=16, num_symbols=10, segment_length=4, right_context_length=2, time_reduction_input_dim=8, time_reduction_stride=2, transformer_num_heads=2, transformer_ffn_dim=32, transformer_num_layers=2,
                   transformer_dropout=0.0, transformer_activation='gelu', transformer_left_context_length=4, transformer_max_memory_size=1, transformer_weight_init_scale_strategy='depthwise', transformer_tanh_on_mem=True,
                   symbol_embedding_dim=8, num_lstm_layers=1, lstm_layer_norm=True, lstm_layer_norm_epsilon=1e-3, lstm_dropout=0.0)
    tgt = torch.randint(0, 10, (1, 4)).float()
    case('rnnt', lambda s, x: s.m(x, s.slen, s.tgt, s.tlen)[0], R(1, 20, D), lambda: dict(m=A.emformer_rnnt_model(**rnnt_kw)), slen=torch.tensor([16]), tgt=tgt.long(), tlen=torch.tensor([4]))
    # ---- SQUIM: objective (waveform -> STOI, PESQ, SI-SDR), subjective (waveform + reference -> MOS) with a tiny SSL model
    case('squim_objective', lambda s, x: torch.stack(s.m(x)).reshape(-1), R(1, 256), lambda: dict(m=A.squim_objective_model(feat_dim=D, win_len=8, d_model=D, nhead=2, hidden_dim=D, num_blocks=1, rnn_type='LSTM', chunk_size=5)))
    import torchaudio.models.squim.subjective as SS
    def subj():
        ssl = A.wav2vec2_model(aux_num_out=None, **w2v); return dict(m=A.SquimSubjective(ssl, nn.Linear(D, 8), SS.Predictor(16, 3)))   # as squim_subjective_model builds it
    case('squim_subjective', lambda s, x: s.m(x, s.ref).reshape(-1), wav, subj, ref=R(1, 400))
    # ---- Tacotron2, teacher forced: tokens + mel spectrogram -> (mel, mel_postnet, gate, alignments)
    tac = dict(n_mels=8, n_symbol=12, symbol_embedding_dim=D, encoder_embedding_dim=D, encoder_n_convolution=2, encoder_kernel_size=3, decoder_rnn_dim=D, attention_rnn_dim=D, attention_hidden_dim=8, attention_location_n_filter=4,
               attention_location_kernel_size=5, prenet_dim=8, postnet_n_convolution=2, postnet_kernel_size=3, postnet_embedding_dim=D, decoder_dropout=0.0, attention_dropout=0.0)
    toks = torch.randint(0, 12, (1, 6)).float()
    import torchaudio.models.tacotron2 as T2M
    class NoPrenetDropout:   # the prenet applies dropout(p=0.5, training=True) even in eval: off for the export and the reference
        def __enter__(c): c.orig = NF.dropout; NF.dropout = lambda x, p=0.5, training=True, inplace=False: x
        def __exit__(c, *a): NF.dropout = c.orig
    def tac_ref(s, x):
        with NoPrenetDropout(): return flat(*s.m(x.long(), s.tlen, s.mel, s.mlen))
    def tac_body(s, x):   # the encoder packs the sequence by lengths (no export path), the decoder masks memory by lengths: full lengths, no mask
        with NoPrenetDropout(): return tac_core(s, x)
    def tac_core(s, x):
        m = s.m; h = m.embedding(x.long()).transpose(1, 2)
        for conv in m.encoder.convolutions: h = NF.relu(conv(h))
        enc, _ = m.encoder.lstm(h.transpose(1, 2))
        orig = T2M._get_mask_from_lengths; T2M._get_mask_from_lengths = lambda lengths: s.nomask
        try: mel, gate, align = m.decoder(enc, s.mel, memory_lengths=s.tlen)
        finally: T2M._get_mask_from_lengths = orig
        return flat(mel, mel + m.postnet(mel), gate, align)
    rcase('tacotron2', tac_body, tac_ref, toks, lambda: dict(m=A.Tacotron2(**tac)), tlen=torch.tensor([6]), mel=R(1, 8, 5), mlen=torch.tensor([5]), nomask=torch.zeros(1, 6, dtype=torch.bool))
    # ---- WaveRNN: waveform + spectrogram -> logits over n_classes per sample
    case('wavernn', lambda s, x: s.m(x, s.spec), R(1, 1, 16), lambda: dict(m=A.WaveRNN(upsample_scales=[2, 2], n_classes=16, hop_length=4, n_res_block=1, n_rnn=16, n_fc=16, kernel_size=3, n_freq=8, n_hidden=8, n_output=8)), spec=R(1, 1, 8, 6))
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
