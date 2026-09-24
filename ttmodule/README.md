# torchtune modules on Hwacha

The modules of meta-pytorch.org/torchtune/0.6/api_ref_modules.html (torchtune 0.6.1), one directory per
entry, run through PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc and checked on Spike against
PyTorch (fixed seed). Each `<case>/` holds `<case>.s` (the assembly of `net`), `mod_main.c` (the generic
host of `../torchfunc`: `net(float* x)` vs the reference with a relative tolerance), `<case>_check.bin`
(input + PyTorch output, `.incbin`) and `HWMLIRFLAGS`. `make` / `make run` / `make <case>` /
`make <case>.spike` / `make gen-<case>` as in `../torchintf`; the cases live in `export_ttf.py`
(needs torchtune in the venv `../.tmenv`: `pip install torchtune==0.6.1 torchao==0.13.0 datasets omegaconf torchdata==0.11.0`).

Every case is a single-input module `net(x)` returning one float tensor: weights, second inputs,
encoder inputs and labels are constant buffers, token ids come in as floats and are cast to long,
losses return their scalar as a 1-element tensor. Sizes are tiny: embed 32, 4 query heads / 2 kv heads
of head_dim 8, sequences of 8, vocab 16. Functions that only configure a model (`get_adapter_params`,
`set_trainable_params`, `register_fusion_module`, `delete_kv_caches` ...) are called on a module,
their effect asserted in `export_ttf.py`, and the module is then run forward as the case.

## Cases: 34 of the page's 34 module entries, all PASS

| section | case | Hwacha |
|---|---|---|
| Modeling components and building blocks | multi_head_attention | PASS, max\|diff\| 0, 4,852 周期 |
| Modeling components and building blocks | feed_forward | PASS, max\|diff\| 0, 2,543 周期 |
| Modeling components and building blocks | kv_cache | PASS, max\|diff\| 0, 284 周期 |
| Modeling components and building blocks | rotary_positional_embeddings | PASS, max\|diff\| 0, 531 周期 |
| Modeling components and building blocks | rmsnorm | PASS, max\|diff\| 0, 583 周期 |
| Modeling components and building blocks | fp32_layer_norm | PASS, max\|diff\| 0, 1,556 周期 |
| Modeling components and building blocks | tanh_gate | PASS, max\|diff\| 0, 102 周期 |
| Modeling components and building blocks | tied_linear | PASS, max\|diff\| 1e-06, 477 周期 |
| Modeling components and building blocks | transformer_self_attention_layer | PASS, max\|diff\| 0, 8,601 周期 |
| Modeling components and building blocks | transformer_cross_attention_layer | PASS, max\|diff\| 0, 7,268 周期 |
| Modeling components and building blocks | transformer_decoder | PASS, max\|diff\| 0, 17,796 周期 |
| Modeling components and building blocks | vision_transformer | PASS, max\|diff\| 0, 12,414 周期 |
| Modeling components and building blocks | layer_dropout | PASS, max\|diff\| 0, 2,543 周期 |
| Modeling components and building blocks | prepare_layer_dropout | PASS, max\|diff\| 0, 4,989 周期 |
| Losses | ce_with_chunked_output_loss | PASS, max\|diff\| 0, 2,379 周期 |
| Losses | forward_kl_loss | PASS, max\|diff\| 0, 2,016 周期 |
| Losses | forward_kl_with_chunked_output_loss | PASS, max\|diff\| 0, 2,016 周期 |
| PEFT components | lora_linear | PASS, max\|diff\| 0, 1,222 周期 |
| PEFT components | dora_linear | PASS, max\|diff\| 0, 2,695 周期 |
| PEFT components | adapter_module | PASS, max\|diff\| 0, 1,014 周期 |
| PEFT components | get_adapter_params | PASS, max\|diff\| 0, 1,222 周期 |
| PEFT components | set_trainable_params | PASS, max\|diff\| 0, 1,222 周期 |
| PEFT components | get_adapter_state_dict | PASS, max\|diff\| 0, 1,222 周期 |
| PEFT components | validate_missing_and_unexpected_for_lora | PASS, max\|diff\| 0, 1,222 周期 |
| PEFT components | disable_adapter | PASS, max\|diff\| 0, 477 周期 |
| Fusion components | deep_fusion_model | PASS, max\|diff\| 0, 25,342 周期 |
| Fusion components | fusion_layer | PASS, max\|diff\| 0, 15,702 周期 |
| Fusion components | fusion_embedding | PASS, max\|diff\| 0, 446 周期 |
| Fusion components | register_fusion_module | PASS, max\|diff\| 0, 7,268 周期 |
| Fusion components | get_fusion_params | PASS, max\|diff\| 0, 17,294 周期 |
| Module utilities | local_kv_cache | PASS, max\|diff\| 0, 17,796 周期 |
| Module utilities | disable_kv_cache | PASS, max\|diff\| 0, 17,367 周期 |
| Module utilities | delete_kv_caches | PASS, max\|diff\| 0, 17,796 周期 |
| Vision transforms | vision_cross_attention_mask | PASS, max\|diff\| 0, 7,792 周期 |

Not cases: the five tokenizer classes and the two tokenizer utilities (string processing, no tensor
computation), `Transform` (a protocol) and `reparametrize_as_dtype_state_dict_post_hook` (a state-dict
hook).

**Exports that differ from the reference** (the reference is always the genuine torchtune call,
asserted equal before export): `kv_cache` writes the update rows with a selection matmul (the cache's
index copy has no lowering); `fusion_embedding` gathers from the concatenated tables by one-hot
(masked_select / masked_scatter have none); the two forward-KL losses are written as the formula
(the module branches on the number of unmasked tokens); `local_kv_cache` runs the cache-free forward
(a full prefill computes the same thing). torchtune registers the RoPE tables and the KV caches with
persistent=False; the export re-registers them as persistent buffers, which torch-mlir's importer
needs ("Could not find state mapping for buffer").
