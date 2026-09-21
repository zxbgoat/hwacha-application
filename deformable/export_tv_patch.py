"""torch-mlir fx importer patch shared with torch-vision/export_tv.py: constant tensors go through numpy
instead of tensor.tolist() (one Python object per element)."""
import numpy as np, torch
import inspect, torch_mlir.extras.fx_importer as _fxi
# torch-mlir turns every constant tensor into a literal via np.array(tensor.tolist()): one Python float
# object per element, ~30x the tensor's size. On the 100M+ parameter models that exhausts memory (it
# took the whole WSL VM down twice). Convert through numpy directly, falling back to the list path.
def _tensor_to_numpy(tensor, npy_dtype):
    try:
        with torch.utils._mode_utils.no_dispatch():
            return np.asarray(tensor.detach().cpu().numpy()).astype(npy_dtype, copy=False)   # asarray keeps 0-d
    except Exception:
        return np.array(tensor.tolist()).astype(npy_dtype)
_src = inspect.getsource(_fxi._make_vtensor_literal_op)
assert "np.array(tensor.tolist()).astype(npy_dtype)" in _src
_fxi._tensor_to_numpy = _tensor_to_numpy
exec(_src.replace("np.array(tensor.tolist()).astype(npy_dtype)", "_tensor_to_numpy(tensor, npy_dtype)"), _fxi.__dict__)
