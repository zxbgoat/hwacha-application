#!/usr/bin/env python3
"""Export one torchtune module (meta-pytorch.org/torchtune/0.6/api_ref_modules.html) through torch-mlir
to linalg-on-tensors, plus a PyTorch reference for one call. Every case is a single-input module
net(x) -> one float tensor (weights, second inputs, labels and encoder inputs are constant buffers;
token ids come in as floats and are cast to long), so the generic host mod_main.c drives them all.
Sizes are tiny (embed 32, 4 heads of 8, sequences of 8).  usage: export_ttf.py <name> <mlir> <bin>"""
import sys, os, struct, math, contextlib, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_mlir import fx
import torchtune.modules as M, torchtune.modules.peft as P, torchtune.modules.loss as LO, torchtune.modules.model_fusion as MF
import torchtune.modules.transforms as TR, torchtune.modules.common_utils as CU
from torchtune.models.clip._position_embeddings import TokenPositionalEmbedding

class Case(nn.Module):
    """net(x) = body(self, x); every tensor in `bufs` becomes a constant buffer, every module in `mods` a submodule"""
    def __init__(s, body, mods=None, **bufs):
        super().__init__(); s.body = body
        for k, v in (mods or {}).items(): setattr(s, k, v)
        for k, v in bufs.items(): s.register_buffer(k, v)
        persist_buffers(s)
    def forward(s, x): return s.body(s, x)
def persist_buffers(root):
    """torchtune registers the RoPE tables and the KV caches with persistent=False; torch-mlir's importer
    needs every buffer in the state dict ("Could not find state mapping for buffer"): re-register them"""
    for m in root.modules():
        for n in list(m._non_persistent_buffers_set):
            t = getattr(m, n); delattr(m, n); m.register_buffer(n, t)
class Ref(Case):
    """exported body differs from the reference (the genuine torchtune call, what check.bin holds):
    the body is an equivalent composition torch-mlir lowers"""
    def __init__(s, body, ref, mods=None, **bufs): super().__init__(body, mods, **bufs); s.ref = ref
    def reference(s, x): return s.ref(s, x)

def R(*shape): return torch.randn(*shape)
D, H, KVH, HD, S, V = 32, 4, 2, 8, 8, 16   # embed, heads, kv heads, head dim, seq, vocab
def rope(): return M.RotaryPositionalEmbeddings(HD, max_seq_len=16)
def mha(causal=True, pos=True, kv=None):
    return M.MultiHeadAttention(embed_dim=D, num_heads=H, num_kv_heads=KVH, head_dim=HD, q_proj=nn.Linear(D, H * HD, bias=False),
                                k_proj=nn.Linear(D, KVH * HD, bias=False), v_proj=nn.Linear(D, KVH * HD, bias=False), output_proj=nn.Linear(D, D, bias=False),
                                pos_embeddings=rope() if pos else None, kv_cache=kv, max_seq_len=16, is_causal=causal)
def ffn(): return M.FeedForward(gate_proj=nn.Linear(D, 2 * D, bias=False), down_proj=nn.Linear(2 * D, D, bias=False), up_proj=nn.Linear(D, 2 * D, bias=False))
def sa_layer(): return M.TransformerSelfAttentionLayer(attn=mha(), mlp=ffn(), sa_norm=M.RMSNorm(D), mlp_norm=M.RMSNorm(D))
def ca_layer(): return M.TransformerCrossAttentionLayer(attn=mha(causal=False, pos=False), mlp=ffn(), ca_norm=M.RMSNorm(D), mlp_norm=M.RMSNorm(D))
def decoder(layers=None):
    return M.TransformerDecoder(tok_embeddings=nn.Embedding(V, D), layers=layers or [sa_layer(), sa_layer()], max_seq_len=16, num_heads=H, head_dim=HD,
                                norm=M.RMSNorm(D), output=nn.Linear(D, V, bias=False))
def causal(n): return torch.tril(torch.ones(n, n, dtype=torch.bool))
def lora(): l = P.LoRALinear(D, V, rank=4, alpha=8.0); nn.init.normal_(l.lora_b.weight, std=0.1); return l   # lora_b is zero-initialised: give it values

def build(name):
    torch.manual_seed(0)
    C = {}
    def case(n, body, x, mods=None, **b): C[n] = lambda: (Case(body, mods, **b), x)
    def rcase(n, body, ref, x, mods=None, **b): C[n] = lambda: (Ref(body, ref, mods, **b), x)
    xs = R(1, S, D); tok = torch.randint(0, V, (1, S)).float()
    # ---- modeling components and building blocks
    case('multi_head_attention', lambda s, x: s.m(x, x), xs, dict(m=mha()))
    case('feed_forward', lambda s, x: s.m(x), xs, dict(m=ffn()))
    # KVCache: a cache holding 4 positions, update() with 2 more (the input: k and v stacked) returns
    # the whole cache; the export writes the rows with a selection matmul instead of an index copy
    k0, v0 = R(1, KVH, 4, HD), R(1, KVH, 4, HD); Psel = torch.zeros(S, 2); Psel[4, 0] = Psel[5, 1] = 1
    base = torch.zeros(2, 1, KVH, S, HD); base[0, :, :, :4] = k0; base[1, :, :, :4] = v0
    def kv_ref(s, x):
        c = M.KVCache(1, S, KVH, HD, torch.float32); c.update(s.k0, s.v0); ko, vo = c.update(x[0], x[1]); return torch.stack([ko, vo])
    rcase('kv_cache', lambda s, x: s.base + s.P @ x, kv_ref, torch.stack([R(1, KVH, 2, HD), R(1, KVH, 2, HD)]), base=base, P=Psel, k0=k0, v0=v0)
    case('rotary_positional_embeddings', lambda s, x: s.m(x), R(1, S, H, HD), dict(m=rope()))
    case('rmsnorm', lambda s, x: s.m(x), xs, dict(m=M.RMSNorm(D)))
    case('fp32_layer_norm', lambda s, x: s.m(x), xs, dict(m=M.Fp32LayerNorm(D)))
    tg = M.TanhGate(); tg.scale.data.fill_(0.5)   # zero-initialised: the gate would output zeros
    case('tanh_gate', lambda s, x: s.m(x), xs, dict(m=tg))
    emb = nn.Embedding(V, D); tl = M.TiedLinear(emb)
    case('tied_linear', lambda s, x: tl(x), xs, dict(emb=emb))
    case('transformer_self_attention_layer', lambda s, x: s.m(x), xs, dict(m=sa_layer()))
    case('transformer_cross_attention_layer', lambda s, x: s.m(x, encoder_input=s.enc), xs, dict(m=ca_layer()), enc=R(1, 6, D))
    case('transformer_decoder', lambda s, x: s.m(x.long()), tok, dict(m=decoder()))
    vit_layer = M.TransformerSelfAttentionLayer(attn=mha(causal=False, pos=False), mlp=ffn(), sa_norm=M.Fp32LayerNorm(D), mlp_norm=M.Fp32LayerNorm(D))
    vit = M.VisionTransformer(patch_size=4, tile_size=8, num_layers=1, embed_dim=D, layer=vit_layer, token_pos_embedding=TokenPositionalEmbedding(D, 4, 8), in_channels=3)
    case('vision_transformer', lambda s, x: s.m(x)[0], R(1, 1, 1, 3, 8, 8), dict(m=vit))
    ld = M.LayerDropout(prob=0.5, disable_on_eval=True)   # eval: the wrapped function runs unconditionally
    case('layer_dropout', lambda s, x: s.ld(s.f, x), xs, dict(f=ffn(), ld=ld))
    layers = nn.ModuleList([ffn(), ffn()]); M.prepare_layer_dropout(layers, prob_max=0.2)
    def run_layers(s, x):
        for l in s.layers: x = l(x)
        return x
    case('prepare_layer_dropout', run_layers, xs, dict(layers=layers))
    # ---- losses (mean-reduced scalars as 1-element tensors; labels carry one ignore_index)
    labels = torch.randint(0, V, (1, S)); labels[0, 3] = -100
    ce = LO.CEWithChunkedOutputLoss(num_output_chunks=2)
    case('ce_with_chunked_output_loss', lambda s, x: ce(list(x.chunk(2, dim=1)), s.lab).reshape(1), R(1, S, V), lab=labels)
    # the KL losses branch on the number of unmasked tokens (`if sum_masks == 0`, data dependent for
    # torch.export): the export writes the formula, -sum_i m_i sum_v p_teacher log p_student / sum_i m_i
    fkl = LO.ForwardKLLoss(); fklc = LO.ForwardKLWithChunkedOutputLoss(num_output_chunks=2)
    def fkl_body(s, x):
        pt = F.softmax(s.t, -1); ls = F.log_softmax(x, -1); m = (s.lab != -100).float()
        return (-((pt * ls).sum(-1) * m).sum() / m.sum()).reshape(1)
    rcase('forward_kl_loss', fkl_body, lambda s, x: fkl(x, s.t, s.lab).reshape(1), R(1, S, V), t=R(1, S, V), lab=labels)
    rcase('forward_kl_with_chunked_output_loss', fkl_body, lambda s, x: fklc(list(x.chunk(2, dim=1)), list(s.t.chunk(2, dim=1)), s.lab).reshape(1), R(1, S, V), t=R(1, S, V), lab=labels)
    # ---- PEFT
    case('lora_linear', lambda s, x: s.m(x), xs, dict(m=lora()))
    dl = P.DoRALinear(D, V, rank=4, alpha=8.0); nn.init.normal_(dl.lora_b.weight, std=0.1); dl.initialize_dora_magnitude()
    case('dora_linear', lambda s, x: s.m(x), xs, dict(m=dl))
    class ScaledAdapter(nn.Module, P.AdapterModule):   # a minimal AdapterModule: y = base(x) + scale * adapter(x)
        def __init__(a):
            super().__init__(); a.base = nn.Linear(D, V, bias=False); a.adapter = nn.Linear(D, V, bias=False); a.scale = nn.Parameter(torch.tensor(0.5))
        def adapter_params(a): return ['adapter.weight', 'scale']
        def forward(a, x): return a.base(x) + a.scale * a.adapter(x)
    sad = ScaledAdapter(); assert set(P.get_adapter_params(sad)) == {'adapter.weight', 'scale'}
    case('adapter_module', lambda s, x: s.m(x), xs, dict(m=sad))
    lm = lora(); ap = P.get_adapter_params(lm); assert set(ap) == {'lora_a.weight', 'lora_b.weight'}
    case('get_adapter_params', lambda s, x: s.m(x), xs, dict(m=lm))
    lm2 = lora(); P.set_trainable_params(lm2, P.get_adapter_params(lm2)); assert not lm2.weight.requires_grad and lm2.lora_a.weight.requires_grad
    case('set_trainable_params', lambda s, x: s.m(x), xs, dict(m=lm2))
    lm3 = lora(); asd = P.get_adapter_state_dict(lm3.state_dict()); assert set(asd) == {'lora_a.weight', 'lora_b.weight'}
    # the adapter state dict alone rebuilds the LoRA delta: base(x) + (alpha / rank) x A^T B^T
    rcase('get_adapter_state_dict', lambda s, x: s.base(x) + 2.0 * (x @ s.A.t() @ s.B.t()), lambda s, x: s.m(x), xs, dict(m=lm3, base=nn.Linear(D, V, bias=False)), A=asd['lora_a.weight'], B=asd['lora_b.weight'])
    C['get_adapter_state_dict'] = (lambda mk: lambda: (lambda m_x: (m_x[0].base.weight.data.copy_(lm3.weight.data), m_x)[1])(mk()))(C['get_adapter_state_dict'])
    lm4 = lora(); P.validate_missing_and_unexpected_for_lora(['q_proj'], False, False, base_missing=['q_proj.lora_a.weight', 'q_proj.lora_b.weight'], base_unexpected=[], lora_missing=['q_proj.weight'], lora_unexpected=[])
    case('validate_missing_and_unexpected_for_lora', lambda s, x: s.m(x), xs, dict(m=lm4))
    lm5 = lora()
    def dis(s, x):
        with P.disable_adapter(s.m): return s.m(x)
    case('disable_adapter', dis, xs, dict(m=lm5))
    # ---- fusion components
    case('fusion_layer', lambda s, x: s.m(x, encoder_input=s.enc), xs, dict(m=MF.FusionLayer(layer=sa_layer(), fusion_layer=ca_layer())), enc=R(1, 6, D))
    fe = MF.FusionEmbedding(V, 4, D); ftok = torch.tensor([[1, 17, 5, 16, 3, 19, 0, 18]]).float()
    # masked_select / masked_scatter have no lowering: the export gathers from [E; E_fusion] by one-hot
    rcase('fusion_embedding', lambda s, x: (x[:, :, None] == s.ids[None, None, :]).float() @ s.tab, lambda s, x: s.m(x.long()), ftok, dict(m=fe),
          ids=torch.arange(V + 4).float(), tab=torch.cat([fe.embedding.weight.data, fe.fusion_embedding.weight.data]))
    class Enc(nn.Module):
        def __init__(e): super().__init__(); e.proj = nn.Linear(D, D, bias=False)
        def forward(e, x): return e.proj(x)
    dfm = MF.DeepFusionModel(decoder=decoder([sa_layer(), MF.FusionLayer(layer=sa_layer(), fusion_layer=ca_layer())]), encoder=Enc())
    case('deep_fusion_model', lambda s, x: s.m(x.long(), encoder_input={'x': s.enc}), tok, dict(m=dfm), enc=R(1, 6, D))
    ca = ca_layer(); MF.register_fusion_module(ca); assert getattr(ca, 'fusion_params')() == list(dict(ca.named_parameters()))
    case('register_fusion_module', lambda s, x: s.m(x, encoder_input=s.enc), xs, dict(m=ca), enc=R(1, 6, D))
    dfm2 = MF.DeepFusionModel(decoder=decoder([MF.FusionLayer(layer=sa_layer(), fusion_layer=ca_layer())]), encoder=Enc())
    fp = MF.get_fusion_params(dfm2); assert fp and all(k.startswith('decoder.layers.0.fusion_layer') for k in fp)
    case('get_fusion_params', lambda s, x: s.m(x.long(), encoder_input={'x': s.enc}), tok, dict(m=dfm2), enc=R(1, 6, D))
    # ---- module utilities (kv-cache contexts around a decoder; positions 0..S-1, causal mask)
    dec = decoder(); pos = torch.arange(S); cmask = torch.zeros(S, 16, dtype=torch.bool); cmask[:, :S] = causal(S)   # against the 16-slot cache
    def with_local(s, x):
        with CU.local_kv_cache(s.m, batch_size=1, device=torch.device('cpu'), dtype=torch.float32, decoder_max_seq_len=16):
            return s.m(x.long(), mask=s.mask[None], input_pos=s.pos)
    # the cached forward index-copies k / v into the cache (no lowering); the export runs the same
    # decoder without caches, which for a full prefill is the same computation
    rcase('local_kv_cache', lambda s, x: s.m(x.long()), with_local, tok, dict(m=dec), mask=cmask, pos=pos)
    dec2 = decoder(); dec2.setup_caches(batch_size=1, dtype=torch.float32, decoder_max_seq_len=16)
    def with_disabled(s, x):
        with CU.disable_kv_cache(s.m): return s.m(x.long())
    case('disable_kv_cache', with_disabled, tok, dict(m=dec2))
    dec3 = decoder(); dec3.setup_caches(batch_size=1, dtype=torch.float32, decoder_max_seq_len=16); CU.delete_kv_caches(dec3); assert not dec3.caches_are_setup()
    case('delete_kv_caches', lambda s, x: s.m(x.long()), tok, dict(m=dec3))
    # ---- vision transforms: the mask transform builds the text -> image attention mask from a token
    # list (constant), applied as the encoder_mask of a cross-attention layer
    vcm = TR.VisionCrossAttentionMask(tile_size=8, patch_size=4, image_token_id=99)
    sample = vcm({'tokens': [99, 5, 6, 7, 8, 9, 10, 11], 'encoder_input': {'images': [torch.zeros(1, 3, 8, 8)]}})
    emask = sample['encoder_mask'][0][None]                      # (1, 8 text tokens, 5 image tokens: 4 patches + CLS)
    case('vision_cross_attention_mask', lambda s, x: s.m(x, encoder_input=s.enc, encoder_mask=s.mask), xs, dict(m=ca_layer()), enc=R(1, 5, D), mask=emask)
    if name == '--list': return sorted(C)
    if name not in C: raise SystemExit('unknown case ' + name)
    return C[name]()

if __name__ == '__main__':
    if sys.argv[1] == '--list': print('\n'.join(build('--list'))); sys.exit(0)
    name, mlir_out, bin_out = sys.argv[1:4]
    m, x = build(name); m = m.eval()
    with torch.no_grad(): y = m.reference(x) if hasattr(m, 'reference') else m(x)
    with torch.no_grad(): assert torch.allclose(m(x), y, atol=1e-4, rtol=1e-4), 'exported body != reference'
    print('%s: in %s -> out %s' % (name, list(x.shape), list(y.shape)))
    mod = fx.export_and_import(m, x, output_type='linalg-on-tensors', func_name='net')
    open(mlir_out, 'w').write(str(mod))
    xf = np.ascontiguousarray(x.numpy()).astype(np.float32).ravel(); yf = np.ascontiguousarray(y.numpy()).astype(np.float32).ravel()
    with open(bin_out, 'wb') as f:
        f.write(struct.pack('i', xf.size)); f.write(xf.tobytes()); f.write(struct.pack('i', yf.size)); f.write(yf.tobytes())
