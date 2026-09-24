# torchtune models on Hwacha

The model families of meta-pytorch.org/torchtune/0.6/api_ref_models.html (torchtune 0.6.1), one case per
architecture / variant, run through PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc and checked on
Spike against PyTorch (fixed seed). Layout, generic host, `gen.sh` and Makefile are those of
`../ttmodule`; the cases live in `export_ttm.py`.

The page's sized builders (`llama2_7b`, `qwen2_5_0_5b`, `llama3_2_vision_11b` ...) are the families'
component builders (`llama2(...)`, `qwen2(...)` ...) with fixed hyperparameters, far beyond Spike. Every
case therefore instantiates the component builder the sized builder calls, at a tiny size (vocab 16,
2 layers, 4 query heads / 2 kv heads of head_dim 8, embed 32, intermediate 64, sequences of 8, images
of one 8x8 tile with 4x4 patches), with the family's distinguishing settings: rope base (code llama,
qwen, phi4), scaled RoPE (llama3.1 / 3.3, factor 32 for 3.2), tied embeddings (llama3.2, qwen2.5),
soft-capping and sliding-window attention (gemma2), one output class (the reward models). LoRA
variants use the `lora_*` builders on q_proj / v_proj and the MLP (rank 4, alpha 8) with `lora_b`
randomised (torchtune zero-initialises it, which would make the adapter the identity). Inputs are
8 token ids (as floats, cast to long) or one image; outputs are the logits (or the reward score).

## Cases: 41, all PASS

| family | case | Hwacha |
|---|---|---|
| llama2 (incl. reward) | llama2 | PASS, max\|diff\| 0, 17,800 周期 |
| llama2 (incl. reward) | lora_llama2 | PASS, max\|diff\| 0, 25,521 周期 |
| llama2 (incl. reward) | llama2_reward | PASS, max\|diff\| 0, 17,828 周期 |
| llama2 (incl. reward) | lora_llama2_reward | PASS, max\|diff\| 0, 25,550 周期 |
| code llama | code_llama2 | PASS, max\|diff\| 0, 17,800 周期 |
| code llama | lora_code_llama2 | PASS, max\|diff\| 0, 25,521 周期 |
| llama3 / 3.1 | llama3 | PASS, max\|diff\| 0, 17,800 周期 |
| llama3 / 3.1 | lora_llama3 | PASS, max\|diff\| 0, 25,521 周期 |
| llama3 / 3.1 | llama3_1 | PASS, max\|diff\| 0, 17,800 周期 |
| llama3 / 3.1 | lora_llama3_1 | PASS, max\|diff\| 0, 25,539 周期 |
| llama3.2 / 3.3 | llama3_2 | PASS, max\|diff\| 7e-06, 17,800 周期 |
| llama3.2 / 3.3 | lora_llama3_2 | PASS, max\|diff\| 5e-06, 25,539 周期 |
| llama3.2 / 3.3 | llama3_3 | PASS, max\|diff\| 0, 17,800 周期 |
| llama3.2 / 3.3 | lora_llama3_3 | PASS, max\|diff\| 0, 25,539 周期 |
| llama3.2 vision | llama3_2_vision | PASS, max\|diff\| 0, 57,606 周期 |
| llama3.2 vision | llama3_2_vision_encoder | PASS, max\|diff\| 0, 31,050 周期 |
| llama3.2 vision | lora_llama3_2_vision_encoder | PASS, max\|diff\| 0, 42,370 周期 |
| llama3.2 vision | llama3_2_vision_decoder | PASS, max\|diff\| 0, 26,668 周期 |
| llama3.2 vision | lora_llama3_2_vision_decoder | PASS, max\|diff\| 0, 38,188 周期 |
| llama3.2 vision | llama3_vision_encoder | PASS, max\|diff\| 0, 31,521 周期 |
| llama3.2 vision | llama3_vision_projection_head | PASS, max\|diff\| 0, 9,921 周期 |
| qwen2 / 2.5 | qwen2 | PASS, max\|diff\| 0, 17,640 周期 |
| qwen2 / 2.5 | lora_qwen2 | PASS, max\|diff\| 0, 25,363 周期 |
| qwen2 / 2.5 | qwen2_5 | PASS, max\|diff\| 3e-06, 17,640 周期 |
| qwen2 / 2.5 | lora_qwen2_5 | PASS, max\|diff\| 3e-06, 25,363 周期 |
| phi3 / 4 | phi3 | PASS, max\|diff\| 0, 17,246 周期 |
| phi3 / 4 | lora_phi3 | PASS, max\|diff\| 0, 24,990 周期 |
| phi3 / 4 | phi4 | PASS, max\|diff\| 0, 17,246 周期 |
| phi3 / 4 | lora_phi4 | PASS, max\|diff\| 0, 24,990 周期 |
| mistral (incl. reward) | mistral | PASS, max\|diff\| 0, 17,800 周期 |
| mistral (incl. reward) | lora_mistral | PASS, max\|diff\| 0, 25,521 周期 |
| mistral (incl. reward) | mistral_reward | PASS, max\|diff\| 0, 17,828 周期 |
| mistral (incl. reward) | lora_mistral_reward | PASS, max\|diff\| 0, 25,550 周期 |
| gemma / gemma2 | gemma | PASS, max\|diff\| 2.9e-05, 18,102 周期 |
| gemma / gemma2 | lora_gemma | PASS, max\|diff\| 4e-05, 25,771 周期 |
| gemma / gemma2 | gemma2 | PASS, max\|diff\| 0.00023, 20,578 周期 |
| gemma / gemma2 | lora_gemma2 | PASS, max\|diff\| 0.00027, 28,316 周期 |
| clip | clip_vision_encoder | PASS, max\|diff\| 0, 21,875 周期 |
| clip | token_positional_embedding | PASS, max\|diff\| 0, 76 周期 |
| clip | tiled_token_positional_embedding | PASS, max\|diff\| 0, 357 周期 |
| clip | tile_positional_embedding | PASS, max\|diff\| 0, 213 周期 |

Not cases: the QLoRA builders (`quantize_base=True` needs torchao's NF4 tensors and its C++ extension,
unavailable for this torch), the tokenizers, chat templates and `llama3_2_vision_transform` (string /
image preprocessing), and the sized builders themselves (same code as the cases, at scale).

**Exports that differ from the reference** (asserted equal before export): the vision decoder's
`FusionEmbedding` (masked_select / masked_scatter have no lowering) is swapped for a one-hot gather from
the concatenated tables in the exported copy; the tiled positional embeddings loop over each image's
aspect ratio (data dependent), so the vision / CLIP encoders use one tile and the two tiled embedding
cases add the constant slices of a fixed (2, 1) aspect ratio. torchtune's non-persistent buffers (RoPE
tables, caches) are re-registered as persistent for torch-mlir's importer.
