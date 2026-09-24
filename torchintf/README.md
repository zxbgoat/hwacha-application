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

## Cases

| function | call | input | output | Hwacha |
|---|---|---|---|---|
| topk | `torch.topk(x, 4)` (dim=-1, largest, sorted) | 4x16 | 4x8 = [values \| indices] per row | PASS, exact |

**topk**: `aten.topk` lowers in torch-mlir to `tm_tensor.sort`, which hwacha-mlir does not take (the
same limit as `fold` / `max_unpool` in `../torchfunc`). The reference is the genuine `torch.topk`; the
exported graph is an equivalent composition of linalg generics: every element's rank in its row is the
number of strictly greater elements (random data, no ties), and the element of rank i is picked by a
one-hot sum, `values_i = sum_j x_j [rank_j == i]`, `indices_i = sum_j j [rank_j == i]`. That is O(N^2)
compares per row instead of a sort, fine at these sizes; the values and the indices are exact.
