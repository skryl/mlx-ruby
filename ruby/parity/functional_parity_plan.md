# Functional Parity Plan (python/mlx -> ruby/lib/mlx)

Goal: 100% functional parity with `python/mlx` behavior, not only API/symbol parity.

Execution rule for every phase:
1. Write phase tests first (red).
2. Implement minimum code to satisfy test expectations (green).
3. Run phase tests and then full suite.
4. Mark phase complete only when both are green.

## Phase Groups

- [x] Phase 149: Schedulers exact step-decay integer semantics and boundary clamping
- [x] Phase 150: Schedulers validation parity (`join_schedules`, `linear_schedule`) and joined-offset behavior
- [x] Phase 151: Scheduler integration with optimizer learning-rate state updates

- [x] Phase 152: Optimizer base state contract (`step`, `learning_rate`, scheduler plumbing)
- [x] Phase 153: Optimizer tree initialization parity (`init`, `init_single` dispatch)
- [x] Phase 154: Optimizer apply/update parity for nested trees
- [x] Phase 155: `MultiOptimizer` split/filter/state parity and `clip_grad_norm`

- [x] Phase 156: `SGD` functional parity (momentum, dampening, nesterov, weight decay)
- [x] Phase 157: `RMSprop` parity
- [x] Phase 158: `Adagrad` + `AdaDelta` parity
- [x] Phase 159: optimizer parameter validation parity

- [x] Phase 160: `Adam` parity (with optional bias correction)
- [x] Phase 161: `AdamW` + `Adamax` parity
- [x] Phase 162: `Lion` parity
- [x] Phase 163: optimizer state serialization compatibility checks

- [x] Phase 164: `Adafactor` parity (factored/unfactored paths)
- [x] Phase 165: `Muon` parity (Newton-Schulz orthogonalization path)
- [x] Phase 166: advanced optimizer integration parity on small model trees

- [x] Phase 167: `nn.Module` dict-backed attribute/state behavior parity
- [x] Phase 168: `Module.parameters`, `trainable_parameters`, freeze/unfreeze parity
- [x] Phase 169: `Module.children`, `leaf_modules`, `modules` traversal parity
- [x] Phase 170: strict update semantics parity (`update`, `update_modules`)
- [x] Phase 171: load/save weights behavior parity

- [x] Phase 172: `nn.utils.value_and_grad` parity
- [x] Phase 173: `nn.utils.checkpoint` parity
- [x] Phase 174: `nn.utils.average_gradients` parity

- [x] Phase 175: losses reduction and validation parity
- [x] Phase 176: `cross_entropy` parity
- [x] Phase 177: `binary_cross_entropy` parity
- [x] Phase 178: `l1_loss`, `mse_loss`, `nll_loss`, `kl_div_loss` parity
- [x] Phase 179: `gaussian_nll_loss` parity
- [x] Phase 180: `smooth_l1_loss`, `triplet_loss` parity
- [x] Phase 181: remaining loss helpers parity

- [x] Phase 182: init base initializers parity (`constant`, `normal`, `uniform`, `identity`)
- [x] Phase 183: fan-in/fan-out helpers parity
- [x] Phase 184: Glorot/He initializers parity
- [x] Phase 185: sparse/orthogonal parity
- [x] Phase 186: initializer validation parity

- [x] Phase 187: `Identity`, `Linear`, `Bilinear` parity
- [ ] Phase 188: `Embedding` + container parity
- [ ] Phase 189: dropout parity
- [ ] Phase 190: activation function/class parity
- [ ] Phase 191: positional encoding parity
- [ ] Phase 192: quantized linear/embedding scaffolding parity

- [ ] Phase 193: convolution layers parity
- [ ] Phase 194: transposed convolution parity
- [ ] Phase 195: pooling parity
- [ ] Phase 196: normalization parity
- [ ] Phase 197: recurrent layers parity
- [ ] Phase 198: transformer layers parity
- [ ] Phase 199: upsample parity
- [ ] Phase 200: distributed layer helpers parity
- [ ] Phase 201: distributed linear classes parity
- [ ] Phase 202: quantized distributed classes parity

- [ ] Phase 203: distributed utils common parity
- [ ] Phase 204: distributed config parity
- [ ] Phase 205: distributed launch parity

- [ ] Phase 206: parity harness for optimizer/loss/module golden cases
- [ ] Phase 207: parity harness for layer golden cases
- [ ] Phase 208: parity harness for distributed-utils golden cases
- [ ] Phase 209: consolidated parity report with uncovered behavior list
- [ ] Phase 210: final cleanup and strict full-suite green gate
