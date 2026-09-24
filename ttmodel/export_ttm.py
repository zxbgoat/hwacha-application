#!/usr/bin/env python3
"""Export one torchtune model architecture (meta-pytorch.org/torchtune/0.6/api_ref_models.html) through
torch-mlir to linalg-on-tensors, plus a PyTorch reference for one call. The sized builders (llama2_7b,
qwen2_5_0_5b ...) are the component builders with fixed hyperparameters and are far beyond Spike, so
every case instantiates the family's component builder at a tiny size (vocab 16, 2 layers, 4 heads /
2 kv heads of head_dim 8, embed 32, intermediate 64, sequences of 8) with the family's distinguishing
settings (rope base, tied embeddings, scale factor, soft-capping ...). Every case is a single-input
module net(x) -> one float tensor (token ids come in as floats and are cast to long; encoder inputs,
masks and aspect ratios are constant buffers).      usage: export_ttm.py <name> <mlir_out> <check_out>"""
import sys, os, struct, numpy as np, torch, torch.nn as nn
from torch_mlir import fx
from torchtune.models import llama2 as L2, llama3 as L3, llama3_1 as L31, llama3_2 as L32, llama3_2_vision as L32V, qwen2 as Q2, phi3 as P3, mistral as MI, gemma as GE, gemma2 as GE2, clip as CL
from torchtune.models.llama2._component_builders import llama2_classifier, lora_llama2_classifier
from torchtune.models.llama3_2_vision._component_builders import llama3_2_vision_projection_head
from torchtune.modules.model_fusion import DeepFusionModel

class Case(nn.Module):
    def __init__(s, body, mods=None, **bufs):
        super().__init__(); s.body = body
        for k, v in (mods or {}).items(): setattr(s, k, v)
        for k, v in bufs.items(): s.register_buffer(k, v)
        persist_buffers(s)
    def forward(s, x): return s.body(s, x)
class Ref(Case):
    """exported body differs from the reference (the genuine torchtune forward, what check.bin holds): the
    body is an equivalent computation torch-mlir lowers; `refmods` are kept out of the exported module"""
    def __init__(s, body, ref, mods=None, refmods=None, **bufs):
        super().__init__(body, mods, **bufs); s.ref = ref; s.refmods = [refmods() if callable(refmods) else refmods]
    def reference(s, x): return s.ref(s, x)
class OneHotFusionEmbedding(nn.Module):
    """FusionEmbedding's forward (masked_select / masked_scatter: no lowering) as a one-hot gather from the
    concatenated tables [E; E_fusion]; same weights, same values"""
    def __init__(s, fe):
        super().__init__()   # the concatenated table as one buffer (a list of Parameters does not import)
        s.register_buffer('tab', torch.cat([fe.embedding.weight.detach(), fe.fusion_embedding.weight.detach()]).clone())
        s.register_buffer('ids', torch.arange(fe.embedding.num_embeddings + fe.fusion_embedding.num_embeddings))
    def forward(s, input): return (input[:, :, None] == s.ids[None, None, :]).to(s.tab.dtype) @ s.tab
def swap_fusion_embeddings(model):
    import copy; from torchtune.modules.model_fusion import FusionEmbedding
    model = copy.deepcopy(model)
    for name, mod in list(model.named_modules()):
        for cn, ch in list(mod.named_children()):
            if isinstance(ch, FusionEmbedding): setattr(mod, cn, OneHotFusionEmbedding(ch))
    return model
def persist_buffers(root):
    """the RoPE tables and KV caches are persistent=False buffers; torch-mlir's importer needs them in the state dict"""
    for m in root.modules():
        for n in list(m._non_persistent_buffers_set):
            t = getattr(m, n); delattr(m, n); m.register_buffer(n, t)
def init_lora(m):
    """torchtune zero-initialises lora_b (the adapter starts as the identity): give it values so the case exercises it"""
    for mod in m.modules():
        if hasattr(mod, 'lora_b'): nn.init.normal_(mod.lora_b.weight, std=0.1)
    return m

V, S, E, I, NL, H, KVH, HD = 16, 8, 32, 64, 2, 4, 2, 8
dec = dict(vocab_size=V, num_layers=NL, num_heads=H, num_kv_heads=KVH, embed_dim=E, max_seq_len=16)
LORA = dict(lora_attn_modules=['q_proj', 'v_proj'], apply_lora_to_mlp=True, lora_rank=4, lora_alpha=8.0)

def build(name):
    torch.manual_seed(0)
    C = {}
    tok = torch.randint(0, V, (1, S)).float()
    def case(n, body, x, mods=None, **b): C[n] = lambda: (Case(body, mods() if callable(mods) else mods, **b), x)   # modules are built lazily
    def rcase(n, body, ref, x, mods=None, refmods=None, **b): C[n] = lambda: (Ref(body, ref, mods() if callable(mods) else mods, refmods, **b), x)
    def decoder(n, mk, **b): case(n, lambda s, x: s.m(x.long()), tok, lambda: dict(m=mk()), **b)
    def vdecoder(n, mk, body, **b):
        """a model with a FusionEmbedding: the export runs the one-hot copy, the reference the genuine model"""
        def both(): r = mk(); return r, swap_fusion_embeddings(r)
        holder = []
        def mods():
            holder[:] = both(); return dict(m=holder[1])
        rcase(n, body, lambda s, x: body(type('o', (), {'m': s.refmods[0], **{k: getattr(s, k) for k in b}})(), x), tok, mods, lambda: holder[0], **b)
    # ---- llama2 family (and code llama: llama2 with rope base 1e6 and its own vocab)
    decoder('llama2', lambda: L2.llama2(intermediate_dim=I, **dec))
    decoder('lora_llama2', lambda: init_lora(L2.lora_llama2(intermediate_dim=I, **LORA, **dec)))
    decoder('llama2_reward', lambda: llama2_classifier(num_classes=1, intermediate_dim=I, **dec))
    decoder('lora_llama2_reward', lambda: init_lora(lora_llama2_classifier(num_classes=1, intermediate_dim=I, **LORA, **dec)))
    decoder('code_llama2', lambda: L2.llama2(intermediate_dim=I, rope_base=1_000_000.0, **dec))
    decoder('lora_code_llama2', lambda: init_lora(L2.lora_llama2(intermediate_dim=I, **LORA, **dec)))   # lora_llama2 has no rope_base argument (torchtune's lora_code_llama2_7b likewise)
    # ---- llama3 / 3.1 (scaled RoPE) / 3.2 (tied embeddings, scale 32) / 3.3 (the 3.1 architecture)
    decoder('llama3', lambda: L3.llama3(intermediate_dim=I, **dec))
    decoder('lora_llama3', lambda: init_lora(L3.lora_llama3(intermediate_dim=I, **LORA, **dec)))
    decoder('llama3_1', lambda: L31.llama3_1(intermediate_dim=I, **dec))
    decoder('lora_llama3_1', lambda: init_lora(L31.lora_llama3_1(intermediate_dim=I, **LORA, **dec)))
    decoder('llama3_2', lambda: L32.llama3_2(intermediate_dim=I, **dec))
    decoder('lora_llama3_2', lambda: init_lora(L32.lora_llama3_2(intermediate_dim=I, **LORA, **dec)))
    decoder('llama3_3', lambda: L31.llama3_1(intermediate_dim=I, scale_factor=8, **dec))
    decoder('lora_llama3_3', lambda: init_lora(L31.lora_llama3_1(intermediate_dim=I, scale_factor=8, **LORA, **dec)))
    # ---- llama3.2 vision: CLIP encoder (tile 8, patch 4: 4 patches + CLS) + projection head, the
    # fusion decoder (a cross-attention fusion layer every 2 layers), and the DeepFusionModel of both
    enc_kw = dict(patch_size=4, num_heads=H, clip_embed_dim=E, clip_num_layers=NL, clip_hidden_states=[0], num_layers_projection=1, decoder_embed_dim=E, tile_size=8, max_num_tiles=1, in_channels=3)
    vdec_kw = dict(vocab_size=V, num_layers=NL, fusion_interval=2, num_special_tokens=2, num_heads=H, num_kv_heads=KVH, embed_dim=E, max_seq_len=16, encoder_max_seq_len=16, intermediate_dim=I)
    img = torch.randn(1, 1, 1, 3, 8, 8)
    case('llama3_2_vision_encoder', lambda s, x: s.m(x), img, lambda: dict(m=L32V.llama3_2_vision_encoder(**enc_kw)))
    case('lora_llama3_2_vision_encoder', lambda s, x: s.m(x), img, lambda: dict(m=init_lora(L32V.lora_llama3_2_vision_encoder(encoder_lora=True, fusion_lora=True, **LORA, **enc_kw))))
    encv = torch.randn(1, 5, E); emask = torch.ones(1, S, 5, dtype=torch.bool)
    vbody = lambda s, x: s.m(x.long(), encoder_input=s.enc, encoder_mask=s.mask)
    vdecoder('llama3_2_vision_decoder', lambda: L32V.llama3_2_vision_decoder(**vdec_kw), vbody, enc=encv, mask=emask)
    vdecoder('lora_llama3_2_vision_decoder', lambda: init_lora(L32V.lora_llama3_2_vision_decoder(decoder_lora=True, fusion_lora=True, **LORA, **vdec_kw)), vbody, enc=encv, mask=emask)
    def vision_model(): return DeepFusionModel(encoder=L32V.llama3_2_vision_encoder(**enc_kw), decoder=L32V.llama3_2_vision_decoder(**vdec_kw))
    vdecoder('llama3_2_vision', vision_model, lambda s, x: s.m(x.long(), encoder_input={'images': s.img}, encoder_mask=s.mask), img=img, mask=emask)
    # the two classes: Llama3VisionEncoder = clip + projection head (what the builder returns), and the
    # projection head alone on a CLIP output with one hidden state
    clipm = lambda: CL.clip_vision_encoder(tile_size=8, patch_size=4, embed_dim=E, num_layers=NL, num_heads=H, out_indices=[0], max_num_tiles=1)
    case('llama3_vision_encoder', lambda s, x: s.m(x), img, lambda: dict(m=L32V.Llama3VisionEncoder(clip=clipm(), projection_head=llama3_2_vision_projection_head(num_layers=1, num_hidden_inputs=1, clip_embed_dim=E, num_heads=H, decoder_embed_dim=E))))
    case('llama3_vision_projection_head', lambda s, x: s.m(x, [s.h]), torch.randn(1, 1, 1, 5, E), lambda: dict(m=llama3_2_vision_projection_head(num_layers=1, num_hidden_inputs=1, clip_embed_dim=E, num_heads=H, decoder_embed_dim=E)), h=torch.randn(1, 1, 1, 5, E))
    # ---- qwen2 / qwen2.5 (qwen2 with tied embeddings, rope 1e6, eps 1e-6, as the qwen2_5 builders do)
    q = dict(intermediate_dim=I, **dec)
    decoder('qwen2', lambda: Q2.qwen2(**q)); decoder('lora_qwen2', lambda: init_lora(Q2.lora_qwen2(**LORA, **q)))
    decoder('qwen2_5', lambda: Q2.qwen2(tie_word_embeddings=True, norm_eps=1e-6, rope_base=1_000_000.0, **q))
    decoder('lora_qwen2_5', lambda: init_lora(Q2.lora_qwen2(tie_word_embeddings=True, norm_eps=1e-6, rope_base=1_000_000.0, **LORA, **q)))
    # ---- phi3 / phi4 (phi3 architecture with the phi4 settings, as phi4_14b does)
    decoder('phi3', lambda: P3.phi3(**q)); decoder('lora_phi3', lambda: init_lora(P3.lora_phi3(**LORA, **q)))
    decoder('phi4', lambda: P3.phi3(rope_base=250_000, **q)); decoder('lora_phi4', lambda: init_lora(P3.lora_phi3(rope_base=250_000, **LORA, **q)))
    # ---- mistral (and the classifier / reward variant: one output class)
    decoder('mistral', lambda: MI.mistral(**q)); decoder('lora_mistral', lambda: init_lora(MI.lora_mistral(**LORA, **q)))
    decoder('mistral_reward', lambda: MI.mistral_classifier(num_classes=1, **q)); decoder('lora_mistral_reward', lambda: init_lora(MI.lora_mistral_classifier(num_classes=1, **LORA, **q)))
    # ---- gemma / gemma2 (soft-capped logits, sliding-window attention on every other layer)
    g = dict(head_dim=HD, intermediate_dim=I, **dec); GL = dict(LORA); GL.pop('apply_lora_to_mlp')
    decoder('gemma', lambda: GE.gemma(**g)); decoder('lora_gemma', lambda: init_lora(GE.lora_gemma(apply_lora_to_mlp=True, **GL, **g)))
    decoder('gemma2', lambda: GE2.gemma2(sliding_window_size=4, query_pre_attn_scalar=HD, **g))
    decoder('lora_gemma2', lambda: init_lora(GE2.lora_gemma2(apply_lora_to_mlp=True, sliding_window_size=4, query_pre_attn_scalar=HD, **GL, **g)))
    # ---- clip: the vision encoder and the three positional embeddings (tile 8, patch 4: 4 patches + CLS)
    case('clip_vision_encoder', lambda s, x: s.m(x)[0], img, lambda: dict(m=CL.clip_vision_encoder(tile_size=8, patch_size=4, embed_dim=E, num_layers=NL, num_heads=H, max_num_tiles=1)))
    case('token_positional_embedding', lambda s, x: s.m(x), torch.randn(1, 1, 1, 5, E), lambda: dict(m=CL.TokenPositionalEmbedding(embed_dim=E, tile_size=8, patch_size=4)))
    # the tiled embeddings loop over the aspect ratio of every image (data dependent for torch.export):
    # for the fixed aspect ratio (2, 1) of a 2-tile image the export adds the same constant slices
    x2 = torch.randn(1, 2, 5, E); ar21 = torch.tensor([[2, 1]])
    rcase('tiled_token_positional_embedding', lambda s, x: x + s.m.local_token_positional_embedding * (1 - s.m.gate.tanh()) + s.m.global_token_positional_embedding[:2, :1].reshape(2, 5, E) * s.m.gate.tanh(),
          lambda s, x: s.m(x.clone(), s.ar), x2, lambda: dict(m=CL.TiledTokenPositionalEmbedding(max_num_tiles=2, embed_dim=E, tile_size=8, patch_size=4)), ar=ar21)
    rcase('tile_positional_embedding', lambda s, x: x + s.m.embedding[:2, :1].reshape(2, 1, E) * s.m.gate.tanh(),
          lambda s, x: s.m(x.clone(), s.ar), x2, lambda: dict(m=CL.TilePositionalEmbedding(max_num_tiles=2, embed_dim=E)), ar=ar21)
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
