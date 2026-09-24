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

## Cases: 34, all PASS

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
