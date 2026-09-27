# torch.* operators on Hwacha

The operator sections of docs.pytorch.org/docs/2.14/torch.html (torch 2.9): Tensors, Creation Ops,
Indexing / Slicing / Joining / Mutating Ops, Random sampling (and the in-place draws), Pointwise Ops,
Reduction Ops, Comparison Ops, Spectral Ops, Other Operations, BLAS and LAPACK Operations and the
Foreach Operations, one case per entry (the in-place `op_` variants included), run through PyTorch ->
torch-mlir -> hwacha-mlir -> hwacha-cc and checked on Spike against PyTorch. Layout, generic host,
`gen.sh` and Makefile are those of `../tafunc`; the cases live in `export_op.py`, the tensor
re-implementations in `op_lib.py`; `probe.sh` exports each case in its own process.

Every case is a single-input module `net(x) -> one float tensor` on a 4x8 input (integer, boolean,
positive or bounded inputs where the op needs them, other shapes where the op dictates): second
operands, indices, masks and weights are constant buffers, integer and boolean results are cast to
float, several results are flattened and concatenated, the in-place variants run on a copy of the
input and return it, complex tensors travel as `[re | im]` along the last dim (torch-mlir has no complex
tensors), and the random-sampling functions draw once under `torch.manual_seed(0)` with the reference
drawn under the same seed. Not on the page but split off as their own cases: `round_decimals`
(`torch.round(x, decimals=1)`).

## Cases: 579, all PASS

| section | case | Hwacha |
|---|---|---|
| Tensors | is_complex | PASS, max\|diff\| 0, 75 周期 |
| Tensors | is_conj | PASS, max\|diff\| 0, 75 周期 |
| Tensors | is_floating_point | PASS, max\|diff\| 0, 72 周期 |
| Tensors | is_nonzero | PASS, max\|diff\| 0, 72 周期 |
| Tensors | is_same_size | PASS, max\|diff\| 0, 72 周期 |
| Tensors | is_signed | PASS, max\|diff\| 0, 72 周期 |
| Tensors | is_tensor | PASS, max\|diff\| 0, 72 周期 |
| Tensors | numel | PASS, max\|diff\| 0, 75 周期 |
| Creation Ops | abs | PASS, max\|diff\| 0, 73 周期 |
| Creation Ops | angle | PASS, max\|diff\| 0, 589 周期 |
| Creation Ops | arange | PASS, max\|diff\| 0, 213 周期 |
| Creation Ops | as_strided | PASS, max\|diff\| 0, 83 周期 |
| Creation Ops | as_tensor | PASS, max\|diff\| 0, 82 周期 |
| Creation Ops | asarray | PASS, max\|diff\| 0, 82 周期 |
| Creation Ops | complex | PASS, max\|diff\| 0, 143 周期 |
| Creation Ops | empty | PASS, max\|diff\| 0, 73 周期 |
| Creation Ops | empty_like | PASS, max\|diff\| 0, 73 周期 |
| Creation Ops | empty_strided | PASS, max\|diff\| 0, 73 周期 |
| Creation Ops | eye | PASS, max\|diff\| 0, 266 周期 |
| Creation Ops | full | PASS, max\|diff\| 0, 75 周期 |
| Creation Ops | full_like | PASS, max\|diff\| 0, 75 周期 |
| Creation Ops | heaviside | PASS, max\|diff\| 0, 379 周期 |
| Creation Ops | imag | PASS, max\|diff\| 0, 87 周期 |
| Creation Ops | linspace | PASS, max\|diff\| 0, 578 周期 |
| Creation Ops | logspace | PASS, max\|diff\| 0, 693 周期 |
| Creation Ops | ones | PASS, max\|diff\| 0, 75 周期 |
| Creation Ops | ones_like | PASS, max\|diff\| 0, 75 周期 |
| Creation Ops | polar | PASS, max\|diff\| 0, 446 周期 |
| Creation Ops | rand | PASS, max\|diff\| 0, 76 周期 |
| Creation Ops | rand_like | PASS, max\|diff\| 0, 76 周期 |
| Creation Ops | randint | PASS, max\|diff\| 0, 76 周期 |
| Creation Ops | randint_like | PASS, max\|diff\| 0, 76 周期 |
| Creation Ops | randn | PASS, max\|diff\| 0, 76 周期 |
| Creation Ops | randn_like | PASS, max\|diff\| 0, 76 周期 |
| Creation Ops | randperm | PASS, max\|diff\| 0, 85 周期 |
| Creation Ops | range | PASS, max\|diff\| 0, 213 周期 |
| Creation Ops | real | PASS, max\|diff\| 0, 85 周期 |
| Creation Ops | scalar_tensor | PASS, max\|diff\| 0, 75 周期 |
| Creation Ops | tensor | PASS, max\|diff\| 0, 82 周期 |
| Creation Ops | zeros | PASS, max\|diff\| 0, 73 周期 |
| Creation Ops | zeros_like | PASS, max\|diff\| 0, 73 周期 |
| Indexing, Slicing, Joining, Mutating Ops | Tensor.expand | PASS, max\|diff\| 0, 79 周期 |
| Indexing, Slicing, Joining, Mutating Ops | Tensor.index_add_ | PASS, max\|diff\| 0, 447 周期 |
| Indexing, Slicing, Joining, Mutating Ops | Tensor.index_copy_ | PASS, max\|diff\| 0, 808 周期 |
| Indexing, Slicing, Joining, Mutating Ops | Tensor.index_reduce_ | PASS, max\|diff\| 0, 752 周期 |
| Indexing, Slicing, Joining, Mutating Ops | Tensor.narrow | PASS, max\|diff\| 0, 87 周期 |
| Indexing, Slicing, Joining, Mutating Ops | Tensor.scatter_ | PASS, max\|diff\| 0, 824 周期 |
| Indexing, Slicing, Joining, Mutating Ops | Tensor.scatter_add_ | PASS, max\|diff\| 0, 623 周期 |
| Indexing, Slicing, Joining, Mutating Ops | Tensor.scatter_reduce_ | PASS, max\|diff\| 0, 850 周期 |
| Indexing, Slicing, Joining, Mutating Ops | adjoint | PASS, max\|diff\| 0, 81 周期 |
| Indexing, Slicing, Joining, Mutating Ops | alias_copy | PASS, max\|diff\| 0, 72 周期 |
| Indexing, Slicing, Joining, Mutating Ops | argwhere | PASS, max\|diff\| 0, 144 周期 |
| Indexing, Slicing, Joining, Mutating Ops | as_strided_copy | PASS, max\|diff\| 0, 83 周期 |
| Indexing, Slicing, Joining, Mutating Ops | as_strided_scatter | PASS, max\|diff\| 0, 162 周期 |
| Indexing, Slicing, Joining, Mutating Ops | cat | PASS, max\|diff\| 0, 122 周期 |
| Indexing, Slicing, Joining, Mutating Ops | chunk | PASS, max\|diff\| 0, 500 周期 |
| Indexing, Slicing, Joining, Mutating Ops | column_stack | PASS, max\|diff\| 0, 143 周期 |
| Indexing, Slicing, Joining, Mutating Ops | concat | PASS, max\|diff\| 0, 143 周期 |
| Indexing, Slicing, Joining, Mutating Ops | concatenate | PASS, max\|diff\| 0, 122 周期 |
| Indexing, Slicing, Joining, Mutating Ops | conj | PASS, max\|diff\| 0, 72 周期 |
| Indexing, Slicing, Joining, Mutating Ops | detach | PASS, max\|diff\| 0, 72 周期 |
| Indexing, Slicing, Joining, Mutating Ops | detach_copy | PASS, max\|diff\| 0, 72 周期 |
| Indexing, Slicing, Joining, Mutating Ops | diagonal | PASS, max\|diff\| 0, 78 周期 |
| Indexing, Slicing, Joining, Mutating Ops | diagonal_copy | PASS, max\|diff\| 0, 78 周期 |
| Indexing, Slicing, Joining, Mutating Ops | diagonal_scatter | PASS, max\|diff\| 0, 162 周期 |
| Indexing, Slicing, Joining, Mutating Ops | dsplit | PASS, max\|diff\| 0, 274 周期 |
| Indexing, Slicing, Joining, Mutating Ops | dstack | PASS, max\|diff\| 0, 139 周期 |
| Indexing, Slicing, Joining, Mutating Ops | expand_copy | PASS, max\|diff\| 0, 79 周期 |
| Indexing, Slicing, Joining, Mutating Ops | fill | PASS, max\|diff\| 0, 75 周期 |
| Indexing, Slicing, Joining, Mutating Ops | gather | PASS, max\|diff\| 0, 130 周期 |
| Indexing, Slicing, Joining, Mutating Ops | hsplit | PASS, max\|diff\| 0, 268 周期 |
| Indexing, Slicing, Joining, Mutating Ops | hstack | PASS, max\|diff\| 0, 143 周期 |
| Indexing, Slicing, Joining, Mutating Ops | index_add | PASS, max\|diff\| 0, 488 周期 |
| Indexing, Slicing, Joining, Mutating Ops | index_copy | PASS, max\|diff\| 0, 808 周期 |
| Indexing, Slicing, Joining, Mutating Ops | index_put_ | PASS, max\|diff\| 0, 608 周期 |
| Indexing, Slicing, Joining, Mutating Ops | index_reduce | PASS, max\|diff\| 0, 1,116 周期 |
| Indexing, Slicing, Joining, Mutating Ops | index_select | PASS, max\|diff\| 0, 88 周期 |
| Indexing, Slicing, Joining, Mutating Ops | masked_fill | PASS, max\|diff\| 0, 151 周期 |
| Indexing, Slicing, Joining, Mutating Ops | masked_select | PASS, max\|diff\| 0, 78 周期 |
| Indexing, Slicing, Joining, Mutating Ops | moveaxis | PASS, max\|diff\| 0, 81 周期 |
| Indexing, Slicing, Joining, Mutating Ops | movedim | PASS, max\|diff\| 0, 81 周期 |
| Indexing, Slicing, Joining, Mutating Ops | narrow | PASS, max\|diff\| 0, 85 周期 |
| Indexing, Slicing, Joining, Mutating Ops | narrow_copy | PASS, max\|diff\| 0, 74 周期 |
| Indexing, Slicing, Joining, Mutating Ops | nonzero | PASS, max\|diff\| 0, 144 周期 |
| Indexing, Slicing, Joining, Mutating Ops | nonzero_static | PASS, max\|diff\| 0, 144 周期 |
| Indexing, Slicing, Joining, Mutating Ops | permute | PASS, max\|diff\| 0, 81 周期 |
| Indexing, Slicing, Joining, Mutating Ops | permute_copy | PASS, max\|diff\| 0, 81 周期 |
| Indexing, Slicing, Joining, Mutating Ops | put | PASS, max\|diff\| 0, 484 周期 |
| Indexing, Slicing, Joining, Mutating Ops | reshape | PASS, max\|diff\| 0, 72 周期 |
| Indexing, Slicing, Joining, Mutating Ops | row_stack | PASS, max\|diff\| 0, 122 周期 |
| Indexing, Slicing, Joining, Mutating Ops | scatter | PASS, max\|diff\| 0, 824 周期 |
| Indexing, Slicing, Joining, Mutating Ops | scatter_add | PASS, max\|diff\| 0, 507 周期 |
| Indexing, Slicing, Joining, Mutating Ops | scatter_reduce | PASS, max\|diff\| 0, 639 周期 |
| Indexing, Slicing, Joining, Mutating Ops | select | PASS, max\|diff\| 0, 77 周期 |
| Indexing, Slicing, Joining, Mutating Ops | select_copy | PASS, max\|diff\| 0, 74 周期 |
| Indexing, Slicing, Joining, Mutating Ops | select_scatter | PASS, max\|diff\| 0, 262 周期 |
| Indexing, Slicing, Joining, Mutating Ops | slice_copy | PASS, max\|diff\| 0, 88 周期 |
| Indexing, Slicing, Joining, Mutating Ops | slice_scatter | PASS, max\|diff\| 0, 126 周期 |
| Indexing, Slicing, Joining, Mutating Ops | split | PASS, max\|diff\| 0, 274 周期 |
| Indexing, Slicing, Joining, Mutating Ops | squeeze | PASS, max\|diff\| 0, 72 周期 |
| Indexing, Slicing, Joining, Mutating Ops | stack | PASS, max\|diff\| 0, 143 周期 |
| Indexing, Slicing, Joining, Mutating Ops | swapaxes | PASS, max\|diff\| 0, 81 周期 |
| Indexing, Slicing, Joining, Mutating Ops | swapdims | PASS, max\|diff\| 0, 81 周期 |
| Indexing, Slicing, Joining, Mutating Ops | t | PASS, max\|diff\| 0, 81 周期 |
| Indexing, Slicing, Joining, Mutating Ops | t_copy | PASS, max\|diff\| 0, 81 周期 |
| Indexing, Slicing, Joining, Mutating Ops | take | PASS, max\|diff\| 0, 78 周期 |
| Indexing, Slicing, Joining, Mutating Ops | take_along_dim | PASS, max\|diff\| 0, 138 周期 |
| Indexing, Slicing, Joining, Mutating Ops | tensor_split | PASS, max\|diff\| 0, 391 周期 |
| Indexing, Slicing, Joining, Mutating Ops | tile | PASS, max\|diff\| 0, 79 周期 |
| Indexing, Slicing, Joining, Mutating Ops | transpose | PASS, max\|diff\| 0, 81 周期 |
| Indexing, Slicing, Joining, Mutating Ops | transpose_copy | PASS, max\|diff\| 0, 81 周期 |
| Indexing, Slicing, Joining, Mutating Ops | unbind | PASS, max\|diff\| 0, 393 周期 |
| Indexing, Slicing, Joining, Mutating Ops | unbind_copy | PASS, max\|diff\| 0, 205 周期 |
| Indexing, Slicing, Joining, Mutating Ops | unravel_index | PASS, max\|diff\| 0, 418 周期 |
| Indexing, Slicing, Joining, Mutating Ops | unsqueeze | PASS, max\|diff\| 0, 72 周期 |
| Indexing, Slicing, Joining, Mutating Ops | unsqueeze_copy | PASS, max\|diff\| 0, 72 周期 |
| Indexing, Slicing, Joining, Mutating Ops | view_copy | PASS, max\|diff\| 0, 72 周期 |
| Indexing, Slicing, Joining, Mutating Ops | vsplit | PASS, max\|diff\| 0, 123 周期 |
| Indexing, Slicing, Joining, Mutating Ops | vstack | PASS, max\|diff\| 0, 122 周期 |
| Indexing, Slicing, Joining, Mutating Ops | where | PASS, max\|diff\| 0, 154 周期 |
| Random sampling | bernoulli | PASS, max\|diff\| 0, 76 周期 |
| Random sampling | multinomial | PASS, max\|diff\| 0, 135 周期 |
| Random sampling | normal | PASS, max\|diff\| 0, 123 周期 |
| Random sampling | poisson | PASS, max\|diff\| 0, 76 周期 |
| In-place random sampling | Tensor.bernoulli_ | PASS, max\|diff\| 0, 121 周期 |
| In-place random sampling | Tensor.cauchy_ | PASS, max\|diff\| 0, 121 周期 |
| In-place random sampling | Tensor.exponential_ | PASS, max\|diff\| 0, 121 周期 |
| In-place random sampling | Tensor.geometric_ | PASS, max\|diff\| 0, 121 周期 |
| In-place random sampling | Tensor.log_normal_ | PASS, max\|diff\| 0, 121 周期 |
| In-place random sampling | Tensor.normal_ | PASS, max\|diff\| 0, 121 周期 |
| In-place random sampling | Tensor.random_ | PASS, max\|diff\| 0, 121 周期 |
| In-place random sampling | Tensor.uniform_ | PASS, max\|diff\| 0, 121 周期 |
| Pointwise Ops | abs_ | PASS, max\|diff\| 0, 73 周期 |
| Pointwise Ops | absolute | PASS, max\|diff\| 0, 73 周期 |
| Pointwise Ops | acos | PASS, max\|diff\| 0, 353 周期 |
| Pointwise Ops | acos_ | PASS, max\|diff\| 0, 353 周期 |
| Pointwise Ops | acosh | PASS, max\|diff\| 0, 292 周期 |
| Pointwise Ops | acosh_ | PASS, max\|diff\| 0, 292 周期 |
| Pointwise Ops | add | PASS, max\|diff\| 0, 78 周期 |
| Pointwise Ops | addcdiv | PASS, max\|diff\| 0, 162 周期 |
| Pointwise Ops | addcmul | PASS, max\|diff\| 0, 162 周期 |
| Pointwise Ops | arccos | PASS, max\|diff\| 0, 353 周期 |
| Pointwise Ops | arccos_ | PASS, max\|diff\| 0, 353 周期 |
| Pointwise Ops | arccosh | PASS, max\|diff\| 0, 292 周期 |
| Pointwise Ops | arccosh_ | PASS, max\|diff\| 0, 292 周期 |
| Pointwise Ops | arcsin | PASS, max\|diff\| 0, 314 周期 |
| Pointwise Ops | arcsin_ | PASS, max\|diff\| 0, 314 周期 |
| Pointwise Ops | arcsinh | PASS, max\|diff\| 0, 291 周期 |
| Pointwise Ops | arcsinh_ | PASS, max\|diff\| 0, 291 周期 |
| Pointwise Ops | arctan | PASS, max\|diff\| 0, 117 周期 |
| Pointwise Ops | arctan2 | PASS, max\|diff\| 0, 531 周期 |
| Pointwise Ops | arctan_ | PASS, max\|diff\| 0, 117 周期 |
| Pointwise Ops | arctanh | PASS, max\|diff\| 0, 322 周期 |
| Pointwise Ops | arctanh_ | PASS, max\|diff\| 0, 322 周期 |
| Pointwise Ops | asin | PASS, max\|diff\| 0, 314 周期 |
| Pointwise Ops | asin_ | PASS, max\|diff\| 0, 314 周期 |
| Pointwise Ops | asinh | PASS, max\|diff\| 0, 291 周期 |
| Pointwise Ops | asinh_ | PASS, max\|diff\| 0, 291 周期 |
| Pointwise Ops | atan | PASS, max\|diff\| 0, 117 周期 |
| Pointwise Ops | atan2 | PASS, max\|diff\| 0, 531 周期 |
| Pointwise Ops | atan_ | PASS, max\|diff\| 0, 117 周期 |
| Pointwise Ops | atanh | PASS, max\|diff\| 0, 322 周期 |
| Pointwise Ops | atanh_ | PASS, max\|diff\| 0, 322 周期 |
| Pointwise Ops | bitwise_and | PASS, max\|diff\| 0, 250 周期 |
| Pointwise Ops | bitwise_left_shift | PASS, max\|diff\| 0, 250 周期 |
| Pointwise Ops | bitwise_not | PASS, max\|diff\| 0, 248 周期 |
| Pointwise Ops | bitwise_or | PASS, max\|diff\| 0, 250 周期 |
| Pointwise Ops | bitwise_right_shift | PASS, max\|diff\| 0, 250 周期 |
| Pointwise Ops | bitwise_xor | PASS, max\|diff\| 0, 250 周期 |
| Pointwise Ops | ceil | PASS, max\|diff\| 0, 157 周期 |
| Pointwise Ops | ceil_ | PASS, max\|diff\| 0, 157 周期 |
| Pointwise Ops | clamp | PASS, max\|diff\| 0, 78 周期 |
| Pointwise Ops | clamp_ | PASS, max\|diff\| 0, 78 周期 |
| Pointwise Ops | clamp_max_ | PASS, max\|diff\| 0, 75 周期 |
| Pointwise Ops | clamp_min_ | PASS, max\|diff\| 0, 76 周期 |
| Pointwise Ops | clip | PASS, max\|diff\| 0, 78 周期 |
| Pointwise Ops | clip_ | PASS, max\|diff\| 0, 78 周期 |
| Pointwise Ops | conj_physical | PASS, max\|diff\| 0, 72 周期 |
| Pointwise Ops | conj_physical_ | PASS, max\|diff\| 0, 72 周期 |
| Pointwise Ops | copysign | PASS, max\|diff\| 0, 290 周期 |
| Pointwise Ops | cos | PASS, max\|diff\| 0, 122 周期 |
| Pointwise Ops | cos_ | PASS, max\|diff\| 0, 122 周期 |
| Pointwise Ops | cosh | PASS, max\|diff\| 0, 338 周期 |
| Pointwise Ops | cosh_ | PASS, max\|diff\| 0, 338 周期 |
| Pointwise Ops | deg2rad | PASS, max\|diff\| 0, 76 周期 |
| Pointwise Ops | deg2rad_ | PASS, max\|diff\| 0, 76 周期 |
| Pointwise Ops | digamma | PASS, max\|diff\| 0, 1,911 周期 |
| Pointwise Ops | div | PASS, max\|diff\| 0, 76 周期 |
| Pointwise Ops | divide | PASS, max\|diff\| 0, 79 周期 |
| Pointwise Ops | erf | PASS, max\|diff\| 0, 133 周期 |
| Pointwise Ops | erf_ | PASS, max\|diff\| 0, 133 周期 |
| Pointwise Ops | erfc | PASS, max\|diff\| 0, 180 周期 |
| Pointwise Ops | erfc_ | PASS, max\|diff\| 0, 180 周期 |
| Pointwise Ops | erfinv | PASS, max\|diff\| 0, 1,918 周期 |
| Pointwise Ops | exp | PASS, max\|diff\| 0, 113 周期 |
| Pointwise Ops | exp2 | PASS, max\|diff\| 0, 161 周期 |
| Pointwise Ops | exp2_ | PASS, max\|diff\| 0, 161 周期 |
| Pointwise Ops | exp_ | PASS, max\|diff\| 0, 113 周期 |
| Pointwise Ops | expm1 | PASS, max\|diff\| 0, 116 周期 |
| Pointwise Ops | expm1_ | PASS, max\|diff\| 0, 116 周期 |
| Pointwise Ops | fake_quantize_per_channel_affine | PASS, max\|diff\| 0, 275 周期 |
| Pointwise Ops | fake_quantize_per_tensor_affine | PASS, max\|diff\| 0, 279 周期 |
| Pointwise Ops | fill_ | PASS, max\|diff\| 0, 68 周期 |
| Pointwise Ops | fix | PASS, max\|diff\| 0, 398 周期 |
| Pointwise Ops | fix_ | PASS, max\|diff\| 0, 398 周期 |
| Pointwise Ops | float_power | PASS, max\|diff\| 1e-06, 257 周期 |
| Pointwise Ops | floor | PASS, max\|diff\| 0, 76 周期 |
| Pointwise Ops | floor_ | PASS, max\|diff\| 0, 76 周期 |
| Pointwise Ops | floor_divide | PASS, max\|diff\| 0, 79 周期 |
| Pointwise Ops | fmod | PASS, max\|diff\| 0, 540 周期 |
| Pointwise Ops | frac | PASS, max\|diff\| 0, 439 周期 |
| Pointwise Ops | frac_ | PASS, max\|diff\| 0, 439 周期 |
| Pointwise Ops | frexp | PASS, max\|diff\| 0, 1,375 周期 |
| Pointwise Ops | gradient | PASS, max\|diff\| 0, 1,151 周期 |
| Pointwise Ops | hypot | PASS, max\|diff\| 0, 220 周期 |
| Pointwise Ops | i0 | PASS, max\|diff\| 0, 3,557 周期 |
| Pointwise Ops | i0_ | PASS, max\|diff\| 0, 3,557 周期 |
| Pointwise Ops | igamma | PASS, max\|diff\| 2e-06, 15,202 周期 |
| Pointwise Ops | igammac | PASS, max\|diff\| 2e-06, 15,240 周期 |
| Pointwise Ops | ldexp | PASS, max\|diff\| 0, 200 周期 |
| Pointwise Ops | ldexp_ | PASS, max\|diff\| 0, 200 周期 |
| Pointwise Ops | lerp | PASS, max\|diff\| 0, 290 周期 |
| Pointwise Ops | lgamma | PASS, max\|diff\| 2e-06, 2,477 周期 |
| Pointwise Ops | log | PASS, max\|diff\| 0, 133 周期 |
| Pointwise Ops | log10 | PASS, max\|diff\| 0, 136 周期 |
| Pointwise Ops | log10_ | PASS, max\|diff\| 0, 136 周期 |
| Pointwise Ops | log1p | PASS, max\|diff\| 0, 135 周期 |
| Pointwise Ops | log1p_ | PASS, max\|diff\| 0, 135 周期 |
| Pointwise Ops | log2 | PASS, max\|diff\| 0, 181 周期 |
| Pointwise Ops | log2_ | PASS, max\|diff\| 0, 181 周期 |
| Pointwise Ops | log_ | PASS, max\|diff\| 0, 133 周期 |
| Pointwise Ops | logaddexp | PASS, max\|diff\| 0, 360 周期 |
| Pointwise Ops | logaddexp2 | PASS, max\|diff\| 0, 478 周期 |
| Pointwise Ops | logical_and | PASS, max\|diff\| 0, 252 周期 |
| Pointwise Ops | logical_not | PASS, max\|diff\| 0, 181 周期 |
| Pointwise Ops | logical_or | PASS, max\|diff\| 0, 252 周期 |
| Pointwise Ops | logical_xor | PASS, max\|diff\| 0, 252 周期 |
| Pointwise Ops | logit | PASS, max\|diff\| 0, 465 周期 |
| Pointwise Ops | logit_ | PASS, max\|diff\| 0, 465 周期 |
| Pointwise Ops | max | PASS, max\|diff\| 0, 76 周期 |
| Pointwise Ops | min | PASS, max\|diff\| 0, 76 周期 |
| Pointwise Ops | mul | PASS, max\|diff\| 0, 76 周期 |
| Pointwise Ops | multiply | PASS, max\|diff\| 0, 76 周期 |
| Pointwise Ops | mvlgamma | PASS, max\|diff\| 4e-06, 5,062 周期 |
| Pointwise Ops | nan_to_num | PASS, max\|diff\| 0, 510 周期 |
| Pointwise Ops | nan_to_num_ | PASS, max\|diff\| 0, 320 周期 |
| Pointwise Ops | neg | PASS, max\|diff\| 0, 73 周期 |
| Pointwise Ops | neg_ | PASS, max\|diff\| 0, 73 周期 |
| Pointwise Ops | negative | PASS, max\|diff\| 0, 73 周期 |
| Pointwise Ops | negative_ | PASS, max\|diff\| 0, 73 周期 |
| Pointwise Ops | nextafter | PASS, max\|diff\| 0, 2,124 周期 |
| Pointwise Ops | nn.functional.softmax | PASS, max\|diff\| 0, 771 周期 |
| Pointwise Ops | polygamma | PASS, max\|diff\| 0, 2,165 周期 |
| Pointwise Ops | positive | PASS, max\|diff\| 0, 72 周期 |
| Pointwise Ops | pow | PASS, max\|diff\| 0, 166 周期 |
| Pointwise Ops | rad2deg | PASS, max\|diff\| 0, 76 周期 |
| Pointwise Ops | rad2deg_ | PASS, max\|diff\| 0, 76 周期 |
| Pointwise Ops | reciprocal | PASS, max\|diff\| 0, 75 周期 |
| Pointwise Ops | reciprocal_ | PASS, max\|diff\| 0, 75 周期 |
| Pointwise Ops | remainder | PASS, max\|diff\| 0, 76 周期 |
| Pointwise Ops | round | PASS, max\|diff\| 0, 82 周期 |
| Pointwise Ops | round_ | PASS, max\|diff\| 0, 82 周期 |
| Pointwise Ops | round_decimals | PASS, max\|diff\| 0, 167 周期 |
| Pointwise Ops | rsqrt | PASS, max\|diff\| 0, 75 周期 |
| Pointwise Ops | rsqrt_ | PASS, max\|diff\| 0, 75 周期 |
| Pointwise Ops | sgn | PASS, max\|diff\| 0, 254 周期 |
| Pointwise Ops | sigmoid | PASS, max\|diff\| 0, 113 周期 |
| Pointwise Ops | sigmoid_ | PASS, max\|diff\| 0, 113 周期 |
| Pointwise Ops | sign | PASS, max\|diff\| 0, 254 周期 |
| Pointwise Ops | signbit | PASS, max\|diff\| 0, 144 周期 |
| Pointwise Ops | sin | PASS, max\|diff\| 0, 122 周期 |
| Pointwise Ops | sin_ | PASS, max\|diff\| 0, 122 周期 |
| Pointwise Ops | sinc | PASS, max\|diff\| 0, 340 周期 |
| Pointwise Ops | sinc_ | PASS, max\|diff\| 0, 340 周期 |
| Pointwise Ops | sinh | PASS, max\|diff\| 0, 338 周期 |
| Pointwise Ops | sinh_ | PASS, max\|diff\| 0, 338 周期 |
| Pointwise Ops | softmax | PASS, max\|diff\| 0, 553 周期 |
| Pointwise Ops | sqrt | PASS, max\|diff\| 0, 73 周期 |
| Pointwise Ops | sqrt_ | PASS, max\|diff\| 0, 73 周期 |
| Pointwise Ops | square | PASS, max\|diff\| 0, 73 周期 |
| Pointwise Ops | square_ | PASS, max\|diff\| 0, 73 周期 |
| Pointwise Ops | sub | PASS, max\|diff\| 0, 78 周期 |
| Pointwise Ops | subtract | PASS, max\|diff\| 0, 76 周期 |
| Pointwise Ops | tan | PASS, max\|diff\| 7e-06, 282 周期 |
| Pointwise Ops | tan_ | PASS, max\|diff\| 7e-06, 282 周期 |
| Pointwise Ops | tanh | PASS, max\|diff\| 0, 115 周期 |
| Pointwise Ops | tanh_ | PASS, max\|diff\| 0, 115 周期 |
| Pointwise Ops | true_divide | PASS, max\|diff\| 0, 76 周期 |
| Pointwise Ops | trunc | PASS, max\|diff\| 0, 398 周期 |
| Pointwise Ops | trunc_ | PASS, max\|diff\| 0, 398 周期 |
| Pointwise Ops | xlogy | PASS, max\|diff\| 0, 363 周期 |
| Pointwise Ops | xlogy_ | PASS, max\|diff\| 0, 363 周期 |
| Pointwise Ops | zero_ | PASS, max\|diff\| 0, 66 周期 |
| Reduction Ops | all | PASS, max\|diff\| 0, 278 周期 |
| Reduction Ops | amax | PASS, max\|diff\| 0, 233 周期 |
| Reduction Ops | amin | PASS, max\|diff\| 0, 342 周期 |
| Reduction Ops | aminmax | PASS, max\|diff\| 0, 548 周期 |
| Reduction Ops | any | PASS, max\|diff\| 0, 277 周期 |
| Reduction Ops | argmax | PASS, max\|diff\| 0, 273 周期 |
| Reduction Ops | argmin | PASS, max\|diff\| 0, 383 周期 |
| Reduction Ops | count_nonzero | PASS, max\|diff\| 0, 272 周期 |
| Reduction Ops | dist | PASS, max\|diff\| 0, 516 周期 |
| Reduction Ops | logsumexp | PASS, max\|diff\| 0, 791 周期 |
| Reduction Ops | mean | PASS, max\|diff\| 0, 187 周期 |
| Reduction Ops | median | PASS, max\|diff\| 0, 857 周期 |
| Reduction Ops | mode | PASS, max\|diff\| 0, 809 周期 |
| Reduction Ops | nanmean | PASS, max\|diff\| 0, 635 周期 |
| Reduction Ops | nanmedian | PASS, max\|diff\| 0, 857 周期 |
| Reduction Ops | nanquantile | PASS, max\|diff\| 0, 1,246 周期 |
| Reduction Ops | nansum | PASS, max\|diff\| 0, 401 周期 |
| Reduction Ops | norm | PASS, max\|diff\| 0, 190 周期 |
| Reduction Ops | norm_except_dim | PASS, max\|diff\| 0, 190 周期 |
| Reduction Ops | nuclear_norm | PASS, max\|diff\| 0, 42,079 周期 |
| Reduction Ops | prod | PASS, max\|diff\| 0, 151 周期 |
| Reduction Ops | quantile | PASS, max\|diff\| 0, 1,246 周期 |
| Reduction Ops | std | PASS, max\|diff\| 0, 587 周期 |
| Reduction Ops | std_mean | PASS, max\|diff\| 0, 882 周期 |
| Reduction Ops | sum | PASS, max\|diff\| 0, 261 周期 |
| Reduction Ops | unique | PASS, max\|diff\| 0, 957 周期 |
| Reduction Ops | unique_consecutive | PASS, max\|diff\| 0, 78 周期 |
| Reduction Ops | var | PASS, max\|diff\| 0, 548 周期 |
| Reduction Ops | var_mean | PASS, max\|diff\| 0, 1,144 周期 |
| Comparison Ops | allclose | PASS, max\|diff\| 0, 641 周期 |
| Comparison Ops | argsort | PASS, max\|diff\| 0, 585 周期 |
| Comparison Ops | eq | PASS, max\|diff\| 0, 147 周期 |
| Comparison Ops | equal | PASS, max\|diff\| 0, 420 周期 |
| Comparison Ops | fmax | PASS, max\|diff\| 0, 295 周期 |
| Comparison Ops | fmin | PASS, max\|diff\| 0, 295 周期 |
| Comparison Ops | ge | PASS, max\|diff\| 0, 147 周期 |
| Comparison Ops | greater | PASS, max\|diff\| 0, 147 周期 |
| Comparison Ops | greater_equal | PASS, max\|diff\| 0, 147 周期 |
| Comparison Ops | gt | PASS, max\|diff\| 0, 147 周期 |
| Comparison Ops | isclose | PASS, max\|diff\| 0, 152 周期 |
| Comparison Ops | isfinite | PASS, max\|diff\| 0, 350 周期 |
| Comparison Ops | isin | PASS, max\|diff\| 0, 377 周期 |
| Comparison Ops | isinf | PASS, max\|diff\| 0, 350 周期 |
| Comparison Ops | isnan | PASS, max\|diff\| 0, 276 周期 |
| Comparison Ops | isneginf | PASS, max\|diff\| 0, 315 周期 |
| Comparison Ops | isposinf | PASS, max\|diff\| 0, 314 周期 |
| Comparison Ops | isreal | PASS, max\|diff\| 0, 68 周期 |
| Comparison Ops | kthvalue | PASS, max\|diff\| 0, 857 周期 |
| Comparison Ops | le | PASS, max\|diff\| 0, 147 周期 |
| Comparison Ops | less | PASS, max\|diff\| 0, 147 周期 |
| Comparison Ops | less_equal | PASS, max\|diff\| 0, 147 周期 |
| Comparison Ops | lt | PASS, max\|diff\| 0, 147 周期 |
| Comparison Ops | maximum | PASS, max\|diff\| 0, 76 周期 |
| Comparison Ops | minimum | PASS, max\|diff\| 0, 76 周期 |
| Comparison Ops | msort | PASS, max\|diff\| 0, 1,012 周期 |
| Comparison Ops | ne | PASS, max\|diff\| 0, 147 周期 |
| Comparison Ops | not_equal | PASS, max\|diff\| 0, 147 周期 |
| Comparison Ops | sort | PASS, max\|diff\| 0, 925 周期 |
| Comparison Ops | topk | PASS, max\|diff\| 0, 1,035 周期 |
| Spectral Ops | bartlett_window | PASS, max\|diff\| 0, 402 周期 |
| Spectral Ops | blackman_window | PASS, max\|diff\| 0, 1,010 周期 |
| Spectral Ops | hamming_window | PASS, max\|diff\| 0, 743 周期 |
| Spectral Ops | hann_window | PASS, max\|diff\| 0, 741 周期 |
| Spectral Ops | istft | PASS, max\|diff\| 2e-06, 8,214 周期 |
| Spectral Ops | kaiser_window | PASS, max\|diff\| 0, 4,313 周期 |
| Spectral Ops | stft | PASS, max\|diff\| 7e-06, 1,729 周期 |
| Other Operations | adaptive_avg_pool1d | PASS, max\|diff\| 0, 163 周期 |
| Other Operations | adaptive_max_pool1d | PASS, max\|diff\| 0, 192 周期 |
| Other Operations | affine_grid_generator | PASS, max\|diff\| 0, 1,272 周期 |
| Other Operations | alpha_dropout | PASS, max\|diff\| 0, 124 周期 |
| Other Operations | alpha_dropout_ | PASS, max\|diff\| 0, 124 周期 |
| Other Operations | atleast_1d | PASS, max\|diff\| 0, 74 周期 |
| Other Operations | atleast_2d | PASS, max\|diff\| 0, 72 周期 |
| Other Operations | atleast_3d | PASS, max\|diff\| 0, 72 周期 |
| Other Operations | avg_pool1d | PASS, max\|diff\| 0, 163 周期 |
| Other Operations | bilinear | PASS, max\|diff\| 0, 575 周期 |
| Other Operations | bincount | PASS, max\|diff\| 0, 1,752 周期 |
| Other Operations | binomial | PASS, max\|diff\| 0, 76 周期 |
| Other Operations | block_diag | PASS, max\|diff\| 0, 367 周期 |
| Other Operations | broadcast_shapes | PASS, max\|diff\| 0, 72 周期 |
| Other Operations | broadcast_tensors | PASS, max\|diff\| 0, 258 周期 |
| Other Operations | broadcast_to | PASS, max\|diff\| 0, 79 周期 |
| Other Operations | bucketize | PASS, max\|diff\| 0, 452 周期 |
| Other Operations | cartesian_prod | PASS, max\|diff\| 0, 264 周期 |
| Other Operations | cdist | PASS, max\|diff\| 0, 350 周期 |
| Other Operations | celu_ | PASS, max\|diff\| 0, 232 周期 |
| Other Operations | channel_shuffle | PASS, max\|diff\| 0, 91 周期 |
| Other Operations | clone | PASS, max\|diff\| 0, 72 周期 |
| Other Operations | combinations | PASS, max\|diff\| 0, 209 周期 |
| Other Operations | conv1d | PASS, max\|diff\| 0, 295 周期 |
| Other Operations | conv3d | PASS, max\|diff\| 2e-06, 5,365 周期 |
| Other Operations | conv_tbc | PASS, max\|diff\| 0, 507 周期 |
| Other Operations | conv_transpose1d | PASS, max\|diff\| 0, 409 周期 |
| Other Operations | conv_transpose2d | PASS, max\|diff\| 0, 684 周期 |
| Other Operations | conv_transpose3d | PASS, max\|diff\| 1e-06, 418 周期 |
| Other Operations | convolution | PASS, max\|diff\| 0, 556 周期 |
| Other Operations | corrcoef | PASS, max\|diff\| 0, 715 周期 |
| Other Operations | cosine_embedding_loss | PASS, max\|diff\| 0, 1,165 周期 |
| Other Operations | cosine_similarity | PASS, max\|diff\| 0, 799 周期 |
| Other Operations | cov | PASS, max\|diff\| 0, 503 周期 |
| Other Operations | cross | PASS, max\|diff\| 0, 692 周期 |
| Other Operations | ctc_loss | PASS, max\|diff\| 0, 10,872 周期 |
| Other Operations | cummax | PASS, max\|diff\| 0, 609 周期 |
| Other Operations | cummin | PASS, max\|diff\| 0, 684 周期 |
| Other Operations | cumprod | PASS, max\|diff\| 0, 615 周期 |
| Other Operations | cumsum | PASS, max\|diff\| 0, 156 周期 |
| Other Operations | detach_ | PASS, max\|diff\| 0, 72 周期 |
| Other Operations | diag | PASS, max\|diff\| 0, 221 周期 |
| Other Operations | diag_embed | PASS, max\|diff\| 0, 226 周期 |
| Other Operations | diagflat | PASS, max\|diff\| 0, 221 周期 |
| Other Operations | diff | PASS, max\|diff\| 0, 89 周期 |
| Other Operations | dropout_ | PASS, max\|diff\| 0, 76 周期 |
| Other Operations | einsum | PASS, max\|diff\| 0, 217 周期 |
| Other Operations | embedding | PASS, max\|diff\| 0, 92 周期 |
| Other Operations | embedding_renorm_ | PASS, max\|diff\| 0, 945 周期 |
| Other Operations | feature_alpha_dropout | PASS, max\|diff\| 0, 585 周期 |
| Other Operations | feature_alpha_dropout_ | PASS, max\|diff\| 0, 585 周期 |
| Other Operations | feature_dropout | PASS, max\|diff\| 0, 363 周期 |
| Other Operations | feature_dropout_ | PASS, max\|diff\| 0, 363 周期 |
| Other Operations | flatten | PASS, max\|diff\| 0, 72 周期 |
| Other Operations | flip | PASS, max\|diff\| 0, 130 周期 |
| Other Operations | fliplr | PASS, max\|diff\| 0, 124 周期 |
| Other Operations | flipud | PASS, max\|diff\| 0, 130 周期 |
| Other Operations | gcd | PASS, max\|diff\| 0, 3,068 周期 |
| Other Operations | gcd_ | PASS, max\|diff\| 0, 3,068 周期 |
| Other Operations | grid_sampler_2d | PASS, max\|diff\| 0, 3,701 周期 |
| Other Operations | grid_sampler_3d | PASS, max\|diff\| 0, 7,555 周期 |
| Other Operations | group_norm | PASS, max\|diff\| 0, 1,372 周期 |
| Other Operations | gru | PASS, max\|diff\| 0, 4,881 周期 |
| Other Operations | gru_cell | PASS, max\|diff\| 0, 1,197 周期 |
| Other Operations | hardshrink | PASS, max\|diff\| 0, 191 周期 |
| Other Operations | hinge_embedding_loss | PASS, max\|diff\| 0, 595 周期 |
| Other Operations | histc | PASS, max\|diff\| 0, 2,302 周期 |
| Other Operations | histogram | PASS, max\|diff\| 0, 2,302 周期 |
| Other Operations | histogramdd | PASS, max\|diff\| 0, 1,332 周期 |
| Other Operations | instance_norm | PASS, max\|diff\| 0, 968 周期 |
| Other Operations | kl_div | PASS, max\|diff\| 0, 1,308 周期 |
| Other Operations | kron | PASS, max\|diff\| 0, 96 周期 |
| Other Operations | lcm | PASS, max\|diff\| 0, 3,169 周期 |
| Other Operations | lcm_ | PASS, max\|diff\| 0, 3,169 周期 |
| Other Operations | logcumsumexp | PASS, max\|diff\| 0, 659 周期 |
| Other Operations | lstm | PASS, max\|diff\| 0, 5,536 周期 |
| Other Operations | lstm_cell | PASS, max\|diff\| 0, 1,411 周期 |
| Other Operations | margin_ranking_loss | PASS, max\|diff\| 0, 479 周期 |
| Other Operations | max_pool1d | PASS, max\|diff\| 0, 192 周期 |
| Other Operations | max_pool3d | PASS, max\|diff\| 0, 147 周期 |
| Other Operations | meshgrid | PASS, max\|diff\| 0, 256 周期 |
| Other Operations | native_batch_norm | PASS, max\|diff\| 0, 378 周期 |
| Other Operations | native_channel_shuffle | PASS, max\|diff\| 0, 89 周期 |
| Other Operations | native_group_norm | PASS, max\|diff\| 0, 1,535 周期 |
| Other Operations | native_layer_norm | PASS, max\|diff\| 0, 1,198 周期 |
| Other Operations | native_norm | PASS, max\|diff\| 0, 291 周期 |
| Other Operations | nn.PairwiseDistance | PASS, max\|diff\| 0, 285 周期 |
| Other Operations | nn.PixelShuffle | PASS, max\|diff\| 0, 86 周期 |
| Other Operations | pairwise_distance | PASS, max\|diff\| 0, 294 周期 |
| Other Operations | pdist | PASS, max\|diff\| 0, 364 周期 |
| Other Operations | pixel_unshuffle | PASS, max\|diff\| 0, 86 周期 |
| Other Operations | poisson_nll_loss | PASS, max\|diff\| 0, 444 周期 |
| Other Operations | prelu | PASS, max\|diff\| 0, 191 周期 |
| Other Operations | ravel | PASS, max\|diff\| 0, 72 周期 |
| Other Operations | relu_ | PASS, max\|diff\| 0, 73 周期 |
| Other Operations | renorm | PASS, max\|diff\| 0, 443 周期 |
| Other Operations | repeat_interleave | PASS, max\|diff\| 0, 87 周期 |
| Other Operations | resize_as_ | PASS, max\|diff\| 0, 72 周期 |
| Other Operations | resolve_conj | PASS, max\|diff\| 0, 72 周期 |
| Other Operations | resolve_neg | PASS, max\|diff\| 0, 72 周期 |
| Other Operations | rms_norm | PASS, max\|diff\| 0, 423 周期 |
| Other Operations | rnn_relu | PASS, max\|diff\| 0, 2,023 周期 |
| Other Operations | rnn_relu_cell | PASS, max\|diff\| 0, 602 周期 |
| Other Operations | rnn_tanh | PASS, max\|diff\| 0, 2,233 周期 |
| Other Operations | rnn_tanh_cell | PASS, max\|diff\| 0, 643 周期 |
| Other Operations | roll | PASS, max\|diff\| 0, 246 周期 |
| Other Operations | rot90 | PASS, max\|diff\| 0, 190 周期 |
| Other Operations | rrelu | PASS, max\|diff\| 0, 192 周期 |
| Other Operations | rrelu_ | PASS, max\|diff\| 0, 192 周期 |
| Other Operations | rsub | PASS, max\|diff\| 0, 78 周期 |
| Other Operations | searchsorted | PASS, max\|diff\| 0, 289 周期 |
| Other Operations | selu | PASS, max\|diff\| 0, 419 周期 |
| Other Operations | selu_ | PASS, max\|diff\| 0, 419 周期 |
| Other Operations | tensordot | PASS, max\|diff\| 0, 161 周期 |
| Other Operations | threshold | PASS, max\|diff\| 0, 153 周期 |
| Other Operations | threshold_ | PASS, max\|diff\| 0, 153 周期 |
| Other Operations | trace | PASS, max\|diff\| 0, 123 周期 |
| Other Operations | tril | PASS, max\|diff\| 0, 337 周期 |
| Other Operations | tril_indices | PASS, max\|diff\| 0, 103 周期 |
| Other Operations | triplet_margin_loss | PASS, max\|diff\| 0, 761 周期 |
| Other Operations | triu | PASS, max\|diff\| 0, 337 周期 |
| Other Operations | triu_indices | PASS, max\|diff\| 0, 103 周期 |
| Other Operations | unflatten | PASS, max\|diff\| 0, 72 周期 |
| Other Operations | vander | PASS, max\|diff\| 0, 236 周期 |
| BLAS and LAPACK Operations | addbmm | PASS, max\|diff\| 0, 287 周期 |
| BLAS and LAPACK Operations | addmm | PASS, max\|diff\| 0, 198 周期 |
| BLAS and LAPACK Operations | addmv | PASS, max\|diff\| 0, 245 周期 |
| BLAS and LAPACK Operations | addmv_ | PASS, max\|diff\| 0, 245 周期 |
| BLAS and LAPACK Operations | addr | PASS, max\|diff\| 0, 139 周期 |
| BLAS and LAPACK Operations | baddbmm | PASS, max\|diff\| 0, 293 周期 |
| BLAS and LAPACK Operations | bmm | PASS, max\|diff\| 0, 178 周期 |
| BLAS and LAPACK Operations | chain_matmul | PASS, max\|diff\| 0, 268 周期 |
| BLAS and LAPACK Operations | cholesky_inverse | PASS, max\|diff\| 0, 6,123 周期 |
| BLAS and LAPACK Operations | cholesky_solve | PASS, max\|diff\| 0, 6,127 周期 |
| BLAS and LAPACK Operations | cumulative_trapezoid | PASS, max\|diff\| 0, 273 周期 |
| BLAS and LAPACK Operations | det | PASS, max\|diff\| 0, 1,091 周期 |
| BLAS and LAPACK Operations | dot | PASS, max\|diff\| 0, 142 周期 |
| BLAS and LAPACK Operations | ger | PASS, max\|diff\| 0, 85 周期 |
| BLAS and LAPACK Operations | inner | PASS, max\|diff\| 0, 228 周期 |
| BLAS and LAPACK Operations | inverse | PASS, max\|diff\| 0, 5,844 周期 |
| BLAS and LAPACK Operations | linalg.det | PASS, max\|diff\| 0, 1,091 周期 |
| BLAS and LAPACK Operations | linalg.inv | PASS, max\|diff\| 0, 5,844 周期 |
| BLAS and LAPACK Operations | linalg.lu_factor | PASS, max\|diff\| 0, 2,679 周期 |
| BLAS and LAPACK Operations | linalg.matrix_exp | PASS, max\|diff\| 0, 3,884 周期 |
| BLAS and LAPACK Operations | linalg.matrix_power | PASS, max\|diff\| 0, 11,617 周期 |
| BLAS and LAPACK Operations | linalg.pinv | PASS, max\|diff\| 0, 5,661 周期 |
| BLAS and LAPACK Operations | linalg.slogdet | PASS, max\|diff\| 0, 2,188 周期 |
| BLAS and LAPACK Operations | lobpcg | PASS, max\|diff\| 0, 42,909 周期 |
| BLAS and LAPACK Operations | logdet | PASS, max\|diff\| 0, 1,175 周期 |
| BLAS and LAPACK Operations | lu | PASS, max\|diff\| 0, 2,679 周期 |
| BLAS and LAPACK Operations | lu_solve | PASS, max\|diff\| 0, 5,965 周期 |
| BLAS and LAPACK Operations | lu_unpack | PASS, max\|diff\| 0, 2,679 周期 |
| BLAS and LAPACK Operations | matmul | PASS, max\|diff\| 0, 161 周期 |
| BLAS and LAPACK Operations | matrix_exp | PASS, max\|diff\| 0, 3,884 周期 |
| BLAS and LAPACK Operations | matrix_power | PASS, max\|diff\| 0, 254 周期 |
| BLAS and LAPACK Operations | mm | PASS, max\|diff\| 0, 161 周期 |
| BLAS and LAPACK Operations | mv | PASS, max\|diff\| 0, 220 周期 |
| BLAS and LAPACK Operations | outer | PASS, max\|diff\| 0, 85 周期 |
| BLAS and LAPACK Operations | pinverse | PASS, max\|diff\| 0, 5,661 周期 |
| BLAS and LAPACK Operations | slogdet | PASS, max\|diff\| 0, 2,190 周期 |
| BLAS and LAPACK Operations | svd | PASS, max\|diff\| 1e-06, 42,021 周期 |
| BLAS and LAPACK Operations | trapezoid | PASS, max\|diff\| 0, 300 周期 |
| BLAS and LAPACK Operations | trapz | PASS, max\|diff\| 0, 338 周期 |
| BLAS and LAPACK Operations | triangular_solve | PASS, max\|diff\| 0, 5,965 周期 |
| BLAS and LAPACK Operations | vdot | PASS, max\|diff\| 0, 142 周期 |
| Foreach Operations | _foreach_abs | PASS, max\|diff\| 0, 448 周期 |
| Foreach Operations | _foreach_abs_ | PASS, max\|diff\| 0, 448 周期 |
| Foreach Operations | _foreach_acos | PASS, max\|diff\| 0, 1,254 周期 |
| Foreach Operations | _foreach_acos_ | PASS, max\|diff\| 0, 1,254 周期 |
| Foreach Operations | _foreach_asin | PASS, max\|diff\| 0, 1,137 周期 |
| Foreach Operations | _foreach_asin_ | PASS, max\|diff\| 0, 1,137 周期 |
| Foreach Operations | _foreach_atan | PASS, max\|diff\| 0, 580 周期 |
| Foreach Operations | _foreach_atan_ | PASS, max\|diff\| 0, 580 周期 |
| Foreach Operations | _foreach_ceil | PASS, max\|diff\| 0, 702 周期 |
| Foreach Operations | _foreach_ceil_ | PASS, max\|diff\| 0, 702 周期 |
| Foreach Operations | _foreach_cos | PASS, max\|diff\| 0, 595 周期 |
| Foreach Operations | _foreach_cos_ | PASS, max\|diff\| 0, 595 周期 |
| Foreach Operations | _foreach_cosh | PASS, max\|diff\| 0, 1,233 周期 |
| Foreach Operations | _foreach_cosh_ | PASS, max\|diff\| 0, 1,233 周期 |
| Foreach Operations | _foreach_erf | PASS, max\|diff\| 0, 628 周期 |
| Foreach Operations | _foreach_erf_ | PASS, max\|diff\| 0, 628 周期 |
| Foreach Operations | _foreach_erfc | PASS, max\|diff\| 0, 771 周期 |
| Foreach Operations | _foreach_erfc_ | PASS, max\|diff\| 0, 771 周期 |
| Foreach Operations | _foreach_exp | PASS, max\|diff\| 0, 568 周期 |
| Foreach Operations | _foreach_exp_ | PASS, max\|diff\| 0, 568 周期 |
| Foreach Operations | _foreach_expm1 | PASS, max\|diff\| 0, 577 周期 |
| Foreach Operations | _foreach_expm1_ | PASS, max\|diff\| 0, 577 周期 |
| Foreach Operations | _foreach_floor | PASS, max\|diff\| 0, 457 周期 |
| Foreach Operations | _foreach_floor_ | PASS, max\|diff\| 0, 457 周期 |
| Foreach Operations | _foreach_frac | PASS, max\|diff\| 0, 1,472 周期 |
| Foreach Operations | _foreach_frac_ | PASS, max\|diff\| 0, 1,472 周期 |
| Foreach Operations | _foreach_lgamma | PASS, max\|diff\| 2e-06, 7,626 周期 |
| Foreach Operations | _foreach_lgamma_ | PASS, max\|diff\| 2e-06, 7,626 周期 |
| Foreach Operations | _foreach_log | PASS, max\|diff\| 0, 628 周期 |
| Foreach Operations | _foreach_log10 | PASS, max\|diff\| 0, 637 周期 |
| Foreach Operations | _foreach_log10_ | PASS, max\|diff\| 0, 637 周期 |
| Foreach Operations | _foreach_log1p | PASS, max\|diff\| 0, 634 周期 |
| Foreach Operations | _foreach_log1p_ | PASS, max\|diff\| 0, 634 周期 |
| Foreach Operations | _foreach_log2 | PASS, max\|diff\| 0, 774 周期 |
| Foreach Operations | _foreach_log2_ | PASS, max\|diff\| 0, 774 周期 |
| Foreach Operations | _foreach_log_ | PASS, max\|diff\| 0, 628 周期 |
| Foreach Operations | _foreach_neg | PASS, max\|diff\| 0, 448 周期 |
| Foreach Operations | _foreach_neg_ | PASS, max\|diff\| 0, 448 周期 |
| Foreach Operations | _foreach_reciprocal | PASS, max\|diff\| 0, 454 周期 |
| Foreach Operations | _foreach_reciprocal_ | PASS, max\|diff\| 0, 454 周期 |
| Foreach Operations | _foreach_round | PASS, max\|diff\| 0, 475 周期 |
| Foreach Operations | _foreach_round_ | PASS, max\|diff\| 0, 475 周期 |
| Foreach Operations | _foreach_sigmoid | PASS, max\|diff\| 0, 568 周期 |
| Foreach Operations | _foreach_sigmoid_ | PASS, max\|diff\| 0, 568 周期 |
| Foreach Operations | _foreach_sin | PASS, max\|diff\| 0, 595 周期 |
| Foreach Operations | _foreach_sin_ | PASS, max\|diff\| 0, 595 周期 |
| Foreach Operations | _foreach_sinh | PASS, max\|diff\| 0, 1,233 周期 |
| Foreach Operations | _foreach_sinh_ | PASS, max\|diff\| 0, 1,233 周期 |
| Foreach Operations | _foreach_sqrt | PASS, max\|diff\| 0, 448 周期 |
| Foreach Operations | _foreach_sqrt_ | PASS, max\|diff\| 0, 448 周期 |
| Foreach Operations | _foreach_tan | PASS, max\|diff\| 0, 1,065 周期 |
| Foreach Operations | _foreach_tan_ | PASS, max\|diff\| 0, 1,065 周期 |
| Foreach Operations | _foreach_trunc | PASS, max\|diff\| 0, 1,355 周期 |
| Foreach Operations | _foreach_trunc_ | PASS, max\|diff\| 0, 1,355 周期 |
| Foreach Operations | _foreach_zero_ | PASS, max\|diff\| 0, 72 周期 |

Not cases (nothing to run on Hwacha): the sections Accelerators, Generators, Serialization,
Parallelism, Locally disabling gradient computation, Utilities, Type Information, Symbolic Numbers,
Export Path, Control Flow (`cond`), Optimizations (`compile`) and Operator Tags; the RNG state functions
(`seed`, `manual_seed`, `initial_seed`, `get/set_rng_state`, `SobolEngine`); the default-dtype /
device / printoptions / flush-denormal setters and `is_storage` / `is_inference` / `is_neg`; the
sparse-tensor constructors, the `*_indices_copy` / `values_copy` / `view_as_complex(_copy)` /
`view_as_real(_copy)` accessors and `resize_as_sparse_`, `dsmm` / `hsmm` / `saddmm` / `spmm` (sparse
layouts have no torch-mlir lowering); `from_file` / `from_numpy` / `from_dlpack` / `frombuffer` (host
memory constructors); the quantized-tensor ops (`quantize_per_*`, `dequantize`, `q_*`, `int_repr`,
`quantized_*`, `fake` moving-average observer, `choose_qparams_optimized`, `fbgemm_*`) and the backend
kernels (`cudnn_*`, `miopen_*`, `mkldnn_*`, the `batch_norm_*` distributed pieces); `empty_permuted` /
`empty_quantized` / `slice_inverse` / `segment_reduce` / `split_copy` / `split_with_sizes_copy` /
`squeeze_copy` / `unfold_copy` / `as_strided_` / `hash_tensor` / `_foreach_clone` (no torch.export path
or no lowering, and duplicates of covered ops); `geqrf` / `orgqr` / `householder_product` / `ormqr`
(Householder QR: `linalg.qr` is covered in `../torchintf`), `svd_lowrank` / `pca_lowrank` (randomised
algorithms on `torch.randn` inside).

## Exports that differ from the reference

The reference is always the genuine torch call, asserted equal to the exported body before export:

- **sort / topk / argsort / msort / kthvalue / median / nanmedian / quantile / nanquantile / mode /
  unique / cummax / cummin**: `aten.sort` / `topk` lower to `tm_tensor.sort`, `cummax` to `tm_tensor.scan`,
  which hwacha-mlir does not take; the exported graphs rank every element by the number of strictly
  greater / smaller elements in its row (ties by index, torch's stable order) and pick by rank with a
  one-hot sum (`op_lib.rank_rows` / `sort_rows` / `quantile_rank`); `mode` counts equal elements;
  `cummax` is a max over the triangular mask; `unique` gathers the first occurrences (a constant) and
  sorts them; `cumsum` / `cumprod` / `logcumsumexp` / `cumulative_trapezoid` are triangular matmuls
  (`cumprod` as exp of the cumsum of logs on a positive input).
- **scatter / scatter_add / scatter_reduce / index_add / index_copy / index_put_ / put / index_reduce /
  bincount / histc / histogram / histogramdd** (`tm_tensor.scatter`): one-hot products with the constant
  index; `index_reduce('prod')` as exp of the summed logs with the sign count; `masked_select`,
  `nonzero`, `argwhere`, `combinations`, `pdist`, `tril/triu_indices`, `unravel_index` have data
  dependent or index-only outputs: the selection is a constant gather, or the case returns the 0/1 mask
  of the positions (the reference converted to the same mask).
- **libm calls hwacha-cc has no vector version of**: `asin` / `acos` (from `atan`), `tan` (sin / cos),
  `sinh` / `cosh` / `asinh` / `acosh` / `atanh` (from `exp` / `log` / `sqrt`), `ceil` (`-floor(-x)`),
  `exp2` / `log2` / `float_power` / `logspace` / `ldexp` / `logaddexp2` (through `exp` / `log`),
  `atan2` (`atan` with the quadrant fixes), `hypot`, `copysign`, `signbit`, `nextafter` (one ulp from the
  exponent), `frexp` (from `log2` and `floor`), `dist` (p = 3 as a power), and the same forms for the
  `_foreach_*` ops (which torch-mlir does not lower at all: the export applies the single-tensor op to
  each list element).
- **special functions** (no torch-mlir lowering): `lgamma` / `mvlgamma`, `digamma`, `polygamma(1, x)`,
  `erfc`, `erfinv`, `i0`, `igamma` / `igammac`: the same shifted asymptotic expansions, Giles'
  approximation, Bessel series and incomplete-gamma series as `../torchintf`'s `torch.special` cases.
- **windows**: `bartlett/blackman/hamming/hann/kaiser_window` have no lowering; the periodic cosine-sum,
  triangle and Kaiser (`I0` series) formulas of `torch.signal.windows`.
- **stft / istft**: as matmuls with the DFT matrix on the reflect-padded, windowed frames; `istft` the
  inverse DFT, window, overlap-add matrix and window-square-sum normalisation (as `../tafunc`).
- **LAPACK**: `inverse` / `linalg.inv` / `cholesky_inverse` / `cholesky_solve` / `lu_solve` /
  `triangular_solve` / `linalg.matrix_power(-2)` by the 4x4 adjugate; `det` / `logdet` / `slogdet` by
  cofactor expansion; `lu` / `lu_factor` / `lu_unpack` as Doolittle without pivoting on an SPD matrix
  (where partial pivoting keeps the identity permutation, asserted); `matrix_exp` by 18 Taylor terms on a
  matrix of norm < 1; `pinverse` / `linalg.pinv` as `(X^T X)^-1 X^T` on a tall full-rank matrix; `svd` /
  `nuclear_norm` through cyclic Jacobi sweeps on `X^T X` (the singular values); `lobpcg` (largest
  eigenvalue) by power iteration, tolerance 1e-3.
- **random sampling**: `rand` / `randn` / `randint` / `randperm` / `bernoulli` / `multinomial` /
  `poisson` / `binomial` / `normal` and the in-place draws, `dropout_` / `alpha_dropout` / `rrelu`: the
  draw under seed 0 is a constant of the case (the RNG ops have no lowering, or the CPU generator is not
  reproducible on Hwacha); `feature_dropout` / `feature_alpha_dropout` and `Tensor.bernoulli_`-free
  cases export the mask arithmetic.
- **1-D pools** (`avg/max/adaptive_*_pool1d`, which lower through `max_pool2d`): reshape + reduce;
  `conv_tbc` as `conv1d` on the permuted input; `grid_sampler_3d` as one-hot trilinear gathers;
  `*_copy` ops as their view forms; `as_strided_scatter` / `diagonal_scatter` / `resize_as_` /
  `fake_quantize_per_channel_affine` written out; `gcd` / `lcm` by a fixed number of Euclid steps on
  floats; `allclose` / `equal` (data-dependent bools) as reductions; `cov` / `corrcoef` / `searchsorted`
  / `vander` / `native_norm` (sparse input) written out; `range` as `arange`; `is_nonzero` on a literal.
- **ctc_loss**: the alpha recursion on the extended label; **embedding_renorm_**: the row scaling
  written out.

`tan`, `stft`, `mvlgamma`, `lgamma` and `istft` differ from the reference by 2e-6 to 7e-6; every other
case by less than 1e-6.
