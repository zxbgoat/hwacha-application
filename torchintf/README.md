# torch.* tensor functions on Hwacha

Top-level tensor functions of docs.pytorch.org/docs/2.14/torch.html, one directory per function, run
through PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc and checked on Spike against PyTorch (fixed
seed). Each `<fn>/` holds `<fn>.s` (the assembly of `net`), `mod_main.c` (the generic host: `net(float* x)`
vs the reference with a relative tolerance), `<fn>_check.bin` (input + PyTorch output, embedded with
`.incbin`) and `HWMLIRFLAGS` (the hwacha-mlir mapping the case needed). The layout, host, `gen.sh` and
Makefile are those of `../torchfunc`.

`make` builds all (`<fn>/<fn>.riscv`), `make run` runs all on Spike, `make <fn>` / `make <fn>.spike` for
one, `make gen-<fn>` regenerates a case from PyTorch (`export_intf.py`, needs the hwacha-cc build tree,
mlir-opt and the torch-mlir venv `../.tmenv`), `make gen-all` every missing one.

Every case is a single-input module `net(x)` returning one float tensor: second operands are constant
buffers, integer results (indices) are cast to float.

## Cases: 75, all PASS

| function | call | input | output | Hwacha |
|---|---|---|---|---|
| topk | `torch.topk(x, 4)` (dim=-1, largest, sorted) | 4x16 | 4x8 = [values \| indices] per row | PASS, exact, 1,116 周期 |
| fft | `torch.fft.fft(x)` | 4x16 | 4x[16\|16] | PASS, max\|diff\| 0, 270 周期 |
| ifft | `torch.fft.ifft(x)` | 4x[16\|16] | 4x[16\|16] | PASS, max\|diff\| 0, 398 周期 |
| fft2 | `torch.fft.fft2(x)` | 2x8x8 | 2x8x[8\|8] | PASS, max\|diff\| 1e-06, 895 周期 |
| ifft2 | `torch.fft.ifft2(x)` | 2x8x[8\|8] | 2x8x[8\|8] | PASS, max\|diff\| 0, 1,008 周期 |
| fftn | `torch.fft.fftn(x)` | 4x4x4 | 4x4x[4\|4] | PASS, max\|diff\| 0, 1,101 周期 |
| ifftn | `torch.fft.ifftn(x)` | 4x4x[4\|4] | 4x4x[4\|4] | PASS, max\|diff\| 0, 1,119 周期 |
| rfft | `torch.fft.rfft(x)` | 4x16 | 4x[9\|9] | PASS, max\|diff\| 0, 274 周期 |
| irfft | `torch.fft.irfft(x, 16)` | 4x[9\|9] | 4x16 | PASS, max\|diff\| 0, 287 周期 |
| rfft2 | `torch.fft.rfft2(x)` | 2x8x8 | 2x8x[5\|5] | PASS, max\|diff\| 1e-06, 838 周期 |
| irfft2 | `torch.fft.irfft2(x, s=(8, 8))` | 2x8x[5\|5] | 2x8x8 | PASS, max\|diff\| 0, 894 周期 |
| rfftn | `torch.fft.rfftn(x)` | 4x4x4 | 4x4x[3\|3] | PASS, max\|diff\| 0, 1,130 周期 |
| irfftn | `torch.fft.irfftn(x, s=(4, 4, 4))` | 4x4x[3\|3] | 4x4x4 | PASS, max\|diff\| 0, 1,131 周期 |
| hfft | `torch.fft.hfft(x, 16)` | 4x[9\|9] | 4x16 | PASS, max\|diff\| 0, 357 周期 |
| ihfft | `torch.fft.ihfft(x)` | 4x16 | 4x[9\|9] | PASS, max\|diff\| 0, 319 周期 |
| hfft2 | `torch.fft.hfft2(x, s=(8, 8))` | 2x8x[5\|5] | 2x8x8 | PASS, max\|diff\| 3e-06, 967 周期 |
| ihfft2 | `torch.fft.ihfft2(x)` | 2x8x8 | 2x8x[5\|5] | PASS, max\|diff\| 0, 889 周期 |
| hfftn | `torch.fft.hfftn(x, s=(4, 4, 4))` | 4x4x[3\|3] | 4x4x4 | PASS, max\|diff\| 1e-06, 1,182 周期 |
| ihfftn | `torch.fft.ihfftn(x)` | 4x4x4 | 4x4x[3\|3] | PASS, max\|diff\| 0, 1,181 周期 |
| fftfreq | `x + torch.fft.fftfreq(16)` | 16 | 16 | PASS, max\|diff\| 0, 76 周期 |
| rfftfreq | `x + torch.fft.rfftfreq(16)` | 9 | 9 | PASS, max\|diff\| 0, 76 周期 |
| fftshift | `torch.fft.fftshift(x)` | 5x7 | 5x7 | PASS, max\|diff\| 0, 171 周期 |
| ifftshift | `torch.fft.ifftshift(x)` | 5x7 | 5x7 | PASS, max\|diff\| 0, 171 周期 |

| bartlett | `x * torch.signal.windows.bartlett(16)` | 4x16 | 4x16 | PASS, max\|diff\| 0, 655 周期 |
| blackman | `x * torch.signal.windows.blackman(16)` | 4x16 | 4x16 | PASS, max\|diff\| 0, 1,008 周期 |
| cosine | `x * torch.signal.windows.cosine(16)` | 4x16 | 4x16 | PASS, max\|diff\| 0, 667 周期 |
| exponential | `x * torch.signal.windows.exponential(16, tau=3.0)` | 4x16 | 4x16 | PASS, max\|diff\| 0, 729 周期 |
| gaussian | `x * torch.signal.windows.gaussian(16, std=3.0)` | 4x16 | 4x16 | PASS, max\|diff\| 0, 731 周期 |
| general_cosine | `x * torch.signal.windows.general_cosine(16, a=[0.42, 0.5, 0.08])` | 4x16 | 4x16 | PASS, max\|diff\| 0, 1,008 周期 |
| general_hamming | `x * torch.signal.windows.general_hamming(16, alpha=0.6)` | 4x16 | 4x16 | PASS, max\|diff\| 0, 959 周期 |
| hamming | `x * torch.signal.windows.hamming(16)` | 4x16 | 4x16 | PASS, max\|diff\| 0, 959 周期 |
| hann | `x * torch.signal.windows.hann(16)` | 4x16 | 4x16 | PASS, max\|diff\| 0, 959 周期 |
| kaiser | `x * torch.signal.windows.kaiser(16, beta=12.0)` | 4x16 | 4x16 | PASS, max\|diff\| 0, 3,781 周期 |
| nuttall | `x * torch.signal.windows.nuttall(16)` | 4x16 | 4x16 | PASS, max\|diff\| 0, 1,033 周期 |

| norm | `torch.linalg.norm(x)` | 4x4 | 1 | PASS, max\|diff\| 0, 211 周期 |
| vector_norm | `torch.linalg.vector_norm(x)` | 4x4 | 1 | PASS, max\|diff\| 0, 211 周期 |
| matrix_norm | `torch.linalg.matrix_norm(x)` | 4x4 | 1 | PASS, max\|diff\| 0, 211 周期 |
| diagonal | `torch.linalg.diagonal(x)` | 4x4 | 4 | PASS, max\|diff\| 0, 77 周期 |
| det | `torch.linalg.det(x)` | 4x4 | 1 | PASS, max\|diff\| 0, 1,862 周期 |
| slogdet | `torch.linalg.slogdet(x)` | 4x4 | [sign, logabsdet] | PASS, max\|diff\| 0, 2,363 周期 |
| cond | `torch.linalg.cond(x)` | 4x4 | 1 | PASS, max\|diff\| 0, 86,910 周期 |
| matrix_rank | `torch.linalg.matrix_rank(x, rtol=1e-3)` | 6x4 (rank 3) | 1 | PASS, max\|diff\| 0, 86,498 周期 |
| cholesky | `torch.linalg.cholesky(x)` | 4x4 SPD | 4x4 | PASS, max\|diff\| 0, 2,178 周期 |
| qr | `torch.linalg.qr(x)` | 4x4 | [Q \| R] 4x8 | PASS, max\|diff\| 0, 3,319 周期 |
| lu | `torch.linalg.lu(x)` | 4x4 | [L \| U] 4x8 (P = I) | PASS, max\|diff\| 0, 2,195 周期 |
| lu_factor | `torch.linalg.lu_factor(x)` | 4x4 | LU 4x4 | PASS, max\|diff\| 0, 1,996 周期 |
| eigh | `torch.linalg.eigh(x)` | 4x4 sym | [V \| w] 4x5 | PASS, max\|diff\| 5e-06, 94,054 周期 |
| eigvalsh | `torch.linalg.eigvalsh(x)` | 4x4 sym | 4 | PASS, max\|diff\| 5e-06, 85,971 周期 |
| eig | `torch.linalg.eig(x)` | 4x4 sym | [V \| Re w] 4x5 | PASS, max\|diff\| 6e-06, 94,054 周期 |
| eigvals | `torch.linalg.eigvals(x)` | 4x4 sym | 4 | PASS, max\|diff\| 6e-06, 85,971 周期 |
| svd | `torch.linalg.svd(x)` | 4x4 | [U \| σ \| Vh] 4x9 | PASS, max\|diff\| 0, 94,842 周期 |
| svdvals | `torch.linalg.svdvals(x)` | 4x4 | 4 | PASS, max\|diff\| 1e-06, 86,076 周期 |
| solve | `torch.linalg.solve(x, B)` | 4x4 | 4x2 | PASS, max\|diff\| 0, 9,164 周期 |
| solve_triangular | `torch.linalg.solve_triangular(x, B, upper=True)` | 4x4 upper | 4x2 | PASS, max\|diff\| 0, 1,113 周期 |
| lu_solve | `torch.linalg.lu_solve(LU, piv, x)` | B 4x2 | 4x2 | PASS, max\|diff\| 0, 2,013 周期 |
| lstsq | `torch.linalg.lstsq(x, B).solution` | 6x4 | 4x3 | PASS, max\|diff\| 0, 9,509 周期 |
| inv | `torch.linalg.inv(x)` | 4x4 | 4x4 | PASS, max\|diff\| 0, 9,049 周期 |
| pinv | `torch.linalg.pinv(x)` | 6x4 | 4x6 | PASS, max\|diff\| 0, 9,375 周期 |
| matrix_exp | `torch.linalg.matrix_exp(x)` | 4x4 | 4x4 | PASS, max\|diff\| 0, 3,248 周期 |
| matrix_power | `torch.linalg.matrix_power(x, 3)` | 4x4 | 4x4 | PASS, max\|diff\| 3e-06, 254 周期 |
| cross | `torch.linalg.cross(x, y)` | 4x3 | 4x3 | PASS, max\|diff\| 0, 434 周期 |
| matmul | `torch.linalg.matmul(x, y)` | 4x4 | 4x4 | PASS, max\|diff\| 0, 136 周期 |
| vecdot | `torch.linalg.vecdot(x, y)` | 4x4 | 4 | PASS, max\|diff\| 0, 198 周期 |
| multi_dot | `torch.linalg.multi_dot([x, y, z])` | 4x4 | 4x4 | PASS, max\|diff\| 1e-06, 259 周期 |
| householder_product | `torch.linalg.householder_product(x, tau)` | 6x3 | 6x3 | PASS, max\|diff\| 0, 1,723 周期 |
| tensorinv | `torch.linalg.tensorinv(x, ind=1)` | 4x2x2 | 2x2x4 | PASS, max\|diff\| 0, 9,049 周期 |
| tensorsolve | `torch.linalg.tensorsolve(x, B)` | 2x2x4 | 4 | PASS, max\|diff\| 0, 9,214 周期 |
| vander | `torch.linalg.vander(x)` | 4 | 4x4 | PASS, max\|diff\| 0, 652 周期 |
| cholesky_ex | `torch.linalg.cholesky_ex(x)[0]` | 4x4 SPD | 4x4 | PASS, max\|diff\| 0, 2,178 周期 |
| inv_ex | `torch.linalg.inv_ex(x)[0]` | 4x4 | 4x4 | PASS, max\|diff\| 0, 9,049 周期 |
| solve_ex | `torch.linalg.solve_ex(x, B)[0]` | 4x4 | 4x2 | PASS, max\|diff\| 0, 9,164 周期 |
| lu_factor_ex | `torch.linalg.lu_factor_ex(x)[0]` | 4x4 | 4x4 | PASS, max\|diff\| 0, 1,996 周期 |
| ldl_factor | `torch.linalg.ldl_factor(x)[0]` | 4x4 SPD | tril(LD) 4x4 | PASS, max\|diff\| 0, 2,218 周期 |
| ldl_factor_ex | `torch.linalg.ldl_factor_ex(x)[0]` | 4x4 SPD | tril(LD) 4x4 | PASS, max\|diff\| 0, 2,218 周期 |
| ldl_solve | `torch.linalg.ldl_solve(LD, piv, x)` | B 4x2 | 4x2 | PASS, max\|diff\| 0, 1,403 周期 |

`[a\|b]`: a complex tensor, real and imaginary parts concatenated along the last dim.

**topk**: `aten.topk` lowers in torch-mlir to `tm_tensor.sort`, which hwacha-mlir does not take (the
same limit as `fold` / `max_unpool` in `../torchfunc`). The reference is the genuine `torch.topk`; the
exported graph is an equivalent composition of linalg generics: every element's rank in its row is the
number of strictly greater elements (random data, no ties), and the element of rank i is picked by a
one-hot sum, `values_i = sum_j x_j [rank_j == i]`, `indices_i = sum_j j [rank_j == i]`. That is O(N^2)
compares per row instead of a sort, fine at these sizes; the values and the indices are exact.

**torch.fft** (docs.pytorch.org/docs/2.14/fft.html, all 22 functions): torch-mlir has no complex
tensors, so a complex tensor is a real one with `[real | imag]` concatenated along the last dim (real
inputs / outputs keep their shape), and a DFT along a dim is a matmul with a constant block matrix:
along the last dim `[xr | xi] @ [[Fr, Fi], [-Fi, Fr]]` (real input: `x @ [Fr | Fi]`), along another dim
`Fr @ x + Fi @ (x @ P)` with `P = [[0, I], [-I, 0]]` turning `[xr | xi]` into `[-xi | xr]`; F = C - iS
(forward) or (C + iS) / N (inverse), C, S = cos / sin(2 pi n k / N), generated in float64. rfft keeps the
first N/2+1 columns, irfft is the C2R sum with the Hermitian weights (1, 2, ..., 2, 1) / N, hfft(x) =
N irfft(conj x), ihfft(x) = conj(rfft x) / N, the 2-D / n-D ones apply the same along every dim (last
dim first for the forward transforms, last for the inverse ones). The reference is the genuine
`torch.fft` call with the complex parts split / joined the same way; `export_intf.py` asserts the two
agree before exporting. `fftfreq` / `rfftfreq` have no torch-mlir lowering (`aten.fft_fftfreq` is
rejected): the vector is a constant buffer added to the input. `fftshift` / `ifftshift` use
`index_select` with constant index vectors instead of `torch.roll` (slice + concat); 5x7 so the two
differ.

**torch.signal.windows** (docs.pytorch.org/docs/2.14/signal.html, all 11 window functions): a window
takes no tensor, so every case windows a signal, `x * window(16)`. The window is computed inside the
exported graph from its definition (torch-mlir lowers the cos / sin / exp / pow / abs, hwacha-cc
expands them on the lanes). `kaiser` needs I0, and `torch.i0` has no lowering: the exported graph
evaluates the power series I0(z) = sum_k (z/2)^2k / (k!)^2 by Horner's rule (30 terms, within 1e-6 of
`torch.i0` at beta = 12); the reference is the genuine `kaiser`.

**torch.linalg** (docs.pytorch.org/docs/2.14/linalg.html, all 41 functions; `intf_linalg.py`):
torch-mlir lowers only the products, `diagonal` and `det`; every factorization and solver is a
fixed-size composition on 4x4 matrices, the reference the genuine call. Inverse: Newton-Schulz
X <- X (2I - A X) from X0 = A^T / (|A|_1 |A|_inf), 30 iterations. LU: Gauss transforms without
pivoting on diagonally dominant inputs (LAPACK does not pivot either, asserted at export). Triangular
solves: (I + N)^-1 = I - N + N^2 - N^3 exactly (N nilpotent). Cholesky / LDL from the unpivoted LU of an
SPD matrix (A = L D L^T, D = diag U). Symmetric eigendecomposition: cyclic Jacobi (10 sweeps of the
6 rotations), eigenvalues sorted with a rank permutation, eigenvector signs fixed by the first
component (the reference likewise); SVD from Jacobi on A^T A (U = A V / sigma); eig / eigvals on a
symmetric input (real spectrum, sorted). QR: modified Gram-Schmidt, the reference's signs normalized
to diag(R) > 0. matrix_exp: scaling and squaring with a 15-term Taylor polynomial. Multiple outputs
are concatenated by matmul with [I 0] / [0 I] buffers (no tensor.concat). The `_ex` variants compare
the result (info is 0, asserted). `norm` / `vector_norm` / `matrix_norm` use sqrt(sum x^2) (the pow
based reductions torch-mlir emits give hwacha-mlir no kernel). Sizes are small (4x4, 6x4), chosen
well conditioned.
