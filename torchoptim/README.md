# torch.optim and torch.nn.init on Hwacha

The interfaces of docs.pytorch.org/docs/2.14/optim.html (torch 2.9): the 16 algorithms (with their
momentum / Nesterov / AMSGrad / centered / maximize variants), the 15 learning-rate schedulers and the
weight-averaging utilities (`AveragedModel` with the SWA / EMA average functions, `SWALR`), plus the
15 initializers of docs.pytorch.org/docs/2.14/nn.init.html (16 cases), one case per interface, run through PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc and checked on Spike
against PyTorch. Layout, generic host, `gen.sh` and Makefile are those of `../tafunc`; the cases live
in `export_opt.py`.

Every case runs K optimizer steps on a small quadratic problem: the parameters are a 4x4 matrix and a
16-vector (the 32-float input), `loss = 1/2 sum w (p - t)^2` with constant `w > 0` and targets `t`, so
the gradient is `w (p - t)` in closed form and the exported graph holds the update rule only
(torch.export does not trace autograd). The exported body calls the algorithm's functional
single-tensor implementation (`torch.optim.<algo>.<algo>`, exactly what `Optimizer.step` runs) K = 4
times; the reference drives the real `Optimizer` on `nn.Parameter`s with `loss.backward()` and the
parameters after the K steps are compared. The scheduler cases run SGD (lr 0.1) for 8 steps with the
scheduler stepped after each: the schedule's learning rates (Python floats the scheduler computes)
are constants of the case, the reference the real `LRScheduler`; `ReduceLROnPlateau` is stepped on
the true loss. The weight-averaging cases output `[parameters | averaged parameters]` after 8 SGD
steps with `AveragedModel.update_parameters` after each (`SWALR` annealing the learning rate to
`swa_lr` over 3 epochs).

## Cases: 57, all PASS

| group | case | interface | Hwacha |
|---|---|---|---|
| algorithm | sgd | `SGD` | PASS, max\|diff\| 0, 1,298 周期 |
| algorithm | sgd_momentum | `SGD(momentum, weight_decay)` | PASS, max\|diff\| 0, 2,398 周期 |
| algorithm | sgd_nesterov | `SGD(nesterov)` | PASS, max\|diff\| 0, 2,409 周期 |
| algorithm | sgd_maximize | `SGD(maximize)` | PASS, max\|diff\| 0, 1,586 周期 |
| algorithm | adam | `Adam` | PASS, max\|diff\| 3e-06, 6,437 周期 |
| algorithm | adam_amsgrad | `Adam(amsgrad, weight_decay)` | PASS, max\|diff\| 3e-06, 7,286 周期 |
| algorithm | adamw | `AdamW` | PASS, max\|diff\| 3e-06, 6,934 周期 |
| algorithm | adamw_amsgrad | `AdamW(amsgrad)` | PASS, max\|diff\| 3e-06, 7,452 周期 |
| algorithm | adamax | `Adamax` | PASS, max\|diff\| 0, 5,315 周期 |
| algorithm | adadelta | `Adadelta` | PASS, max\|diff\| 0, 6,143 周期 |
| algorithm | adagrad | `Adagrad` | PASS, max\|diff\| 0, 3,681 周期 |
| algorithm | nadam | `NAdam` | PASS, max\|diff\| 2e-06, 7,690 周期 |
| algorithm | radam | `RAdam` | PASS, max\|diff\| 0, 3,568 周期 |
| algorithm | rmsprop | `RMSprop` | PASS, max\|diff\| 0, 4,384 周期 |
| algorithm | rmsprop_centered | `RMSprop(centered, momentum)` | PASS, max\|diff\| 0, 7,440 周期 |
| algorithm | rprop | `Rprop` | PASS, max\|diff\| 0, 8,982 周期 |
| algorithm | asgd | `ASGD` | PASS, max\|diff\| 0, 2,228 周期 |
| algorithm | adafactor | `Adafactor` | PASS, max\|diff\| 0, 31,199 周期 |
| algorithm | muon | `Muon` | PASS, max\|diff\| 0, 38,523 周期 |
| algorithm | sparse_adam | `SparseAdam` | PASS, max\|diff\| 0, 5,392 周期 |
| algorithm | lbfgs | `LBFGS` | PASS, max\|diff\| 0, 24,160 周期 |
| lr scheduler | lambda_lr | `lr_scheduler.LambdaLR` | PASS, max\|diff\| 0, 2,466 周期 |
| lr scheduler | multiplicative_lr | `lr_scheduler.MultiplicativeLR` | PASS, max\|diff\| 0, 2,466 周期 |
| lr scheduler | step_lr | `lr_scheduler.StepLR` | PASS, max\|diff\| 0, 2,466 周期 |
| lr scheduler | multi_step_lr | `lr_scheduler.MultiStepLR` | PASS, max\|diff\| 0, 2,466 周期 |
| lr scheduler | constant_lr | `lr_scheduler.ConstantLR` | PASS, max\|diff\| 0, 2,466 周期 |
| lr scheduler | linear_lr | `lr_scheduler.LinearLR` | PASS, max\|diff\| 0, 2,464 周期 |
| lr scheduler | exponential_lr | `lr_scheduler.ExponentialLR` | PASS, max\|diff\| 0, 2,466 周期 |
| lr scheduler | polynomial_lr | `lr_scheduler.PolynomialLR` | PASS, max\|diff\| 0, 2,454 周期 |
| lr scheduler | cosine_annealing_lr | `lr_scheduler.CosineAnnealingLR` | PASS, max\|diff\| 0, 2,466 周期 |
| lr scheduler | chained_scheduler | `lr_scheduler.ChainedScheduler` | PASS, max\|diff\| 0, 2,466 周期 |
| lr scheduler | sequential_lr | `lr_scheduler.SequentialLR` | PASS, max\|diff\| 0, 2,466 周期 |
| lr scheduler | cyclic_lr | `lr_scheduler.CyclicLR` | PASS, max\|diff\| 0, 2,466 周期 |
| lr scheduler | one_cycle_lr | `lr_scheduler.OneCycleLR` | PASS, max\|diff\| 0, 2,466 周期 |
| lr scheduler | cosine_annealing_warm_restarts | `lr_scheduler.CosineAnnealingWarmRestarts` | PASS, max\|diff\| 0, 2,466 周期 |
| lr scheduler | reduce_lr_on_plateau | `lr_scheduler.ReduceLROnPlateau` | PASS, max\|diff\| 0, 2,466 周期 |
| weight averaging | averaged_model_swa | `swa_utils.AveragedModel + get_swa_avg_fn` | PASS, max\|diff\| 0, 4,653 周期 |
| weight averaging | averaged_model_ema | `swa_utils.AveragedModel + get_ema_avg_fn` | PASS, max\|diff\| 0, 4,990 周期 |
| weight averaging | swa_multi_avg_fn | `swa_utils.get_swa_multi_avg_fn` | PASS, max\|diff\| 0, 4,653 周期 |
| weight averaging | ema_multi_avg_fn | `swa_utils.get_ema_multi_avg_fn` | PASS, max\|diff\| 0, 4,990 周期 |
| weight averaging | swalr | `swa_utils.SWALR (+ AveragedModel)` | PASS, max\|diff\| 0, 4,653 周期 |
| nn.init | calculate_gain | `nn.init.calculate_gain (leaky_relu 0.2, tanh, relu, selu, linear)` | PASS, max\|diff\| 0, 239 周期 |
| nn.init | constant_ | `nn.init.constant_(0.3)` | PASS, max\|diff\| 0, 69 周期 |
| nn.init | ones_ | `nn.init.ones_` | PASS, max\|diff\| 0, 68 周期 |
| nn.init | zeros_ | `nn.init.zeros_` | PASS, max\|diff\| 0, 66 周期 |
| nn.init | eye_ | `nn.init.eye_` | PASS, max\|diff\| 0, 274 周期 |
| nn.init | dirac_ | `nn.init.dirac_ (4x2x3x3)` | PASS, max\|diff\| 0, 145 周期 |
| nn.init | dirac_groups | `nn.init.dirac_(groups=2) (4x2x3)` | PASS, max\|diff\| 0, 141 周期 |
| nn.init | uniform_ | `nn.init.uniform_(-0.5, 0.5)` | PASS, max\|diff\| 0, 121 周期 |
| nn.init | normal_ | `nn.init.normal_(0, 0.5)` | PASS, max\|diff\| 0, 121 周期 |
| nn.init | trunc_normal_ | `nn.init.trunc_normal_(0, 1, -1.5, 1.5)` | PASS, max\|diff\| 0, 121 周期 |
| nn.init | xavier_uniform_ | `nn.init.xavier_uniform_(gain=1.5)` | PASS, max\|diff\| 0, 121 周期 |
| nn.init | xavier_normal_ | `nn.init.xavier_normal_` | PASS, max\|diff\| 0, 121 周期 |
| nn.init | kaiming_uniform_ | `nn.init.kaiming_uniform_(a=0.1, fan_in, leaky_relu)` | PASS, max\|diff\| 0, 121 周期 |
| nn.init | kaiming_normal_ | `nn.init.kaiming_normal_(fan_out, relu)` | PASS, max\|diff\| 0, 121 周期 |
| nn.init | sparse_ | `nn.init.sparse_(0.5, std=0.1)` | PASS, max\|diff\| 0, 121 周期 |
| nn.init | orthogonal_ | `nn.init.orthogonal_(gain=1.5)` | PASS, max\|diff\| 0, 6,047 周期 |

Not cases: the `Optimizer` base-class methods and hooks (`step`, `zero_grad`, `add_param_group`,
`state_dict` / `load_state_dict` and their pre / post hooks, `register_optimizer_step_pre/post_hook`,
`swap_in_optimizer_params_and_state`), `LRScheduler` (the base class), and `swa_utils.update_bn`:
Python-side bookkeeping, not tensor computations. `Optimizer.step` is what every algorithm case runs.

## Notes

- The in-place functional updates (`param.add_`, `exp_avg.lerp_`, ..) run on copies of the input
  slices: on views of the input they export as input mutations and the host reads back the mutated
  input instead of the result.
- Adafactor: `_single_tensor_adafactor` takes the parameter's RMS (for `alpha`) and the update's RMS
  (for the clipping) as Python floats through `.item()`, which torch.export cannot resolve; the body
  is the same algorithm with those kept as tensors.
- Rprop: the sign bookkeeping's masked assignments (`sign[sign > 0] = etaplus`, `grad[sign == etaminus]
  = 0`) lower to `tm_tensor.scan`; the body writes them as `torch.where` (what the `capturable=True`
  path of the same function does, which only runs on accelerators).
- Muon: the Newton-Schulz orthogonalisation runs in bfloat16 (`grad.bfloat16()`), which the RISC-V
  toolchain has no support for (`__truncsfbf2`); both the export and the reference run it in float32
  (the same iteration, the cast dropped). The vector parameter is given as a 1x16 matrix (Muon takes
  2-D parameters).
- SparseAdam: the Adam update on the rows a sparse gradient touches; on the dense problem every row is
  touched, so the body is `torch.optim._functional.sparse_adam`'s math written densely, the reference
  the real `SparseAdam` fed `grad.to_sparse()`.
- LBFGS: `LBFGS.step` (lr 0.1, max_iter 3, history 4, no line search, tolerances 0) unrolled over the 4
  closure calls with its global state (direction, step, history, previous gradient, iteration count)
  carried across them; `t = min(1, 1 / |g|_1) * lr` on the very first iteration, `lr` afterwards; the
  `ys > 1e-10` history-update condition holds on this problem.
- AveragedModel's first `update_parameters` copies the parameters (`n_averaged == 0`); the averaging
  starts at the second.
- nn.init: every initializer fills a tensor in place; the case runs it on a copy of the 4x8 input (the
  shapes `dirac_` needs otherwise) and returns the filled tensor. The random initializers (`uniform_`,
  `normal_`, `trunc_normal_`, `xavier_*`, `kaiming_*`, `sparse_`) draw under `torch.manual_seed(0)`; the
  RNG ops have no lowering / the CPU generator is not reproduced on Hwacha, so the seed-0 draw is a
  constant of the case and the reference redraws it. `dirac_` / `eye_` assign by index: the delta
  pattern is a constant. `orthogonal_` is the QR of the seed-0 Gaussian (`linalg.qr` has no lowering):
  Gram-Schmidt on the pinned Gaussian, the column signs fixed by `sign(diag(R))` as torch does.
- Adam / AdamW / NAdam differ from the reference by up to 3e-6: the bias-corrected step sizes are
  Python floats in the functional implementation and float32 products on Hwacha.
