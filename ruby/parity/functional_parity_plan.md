# Functional Parity Plan (python/mlx -> ruby/lib/mlx)

Goal: 100% functional parity with `python/mlx` behavior, not only API/symbol parity.

Execution rule for every phase:
1. Write phase tests first (red).
2. Implement minimum code to satisfy test expectations (green).
3. Run phase tests and then full suite.
4. Mark phase complete only when both are green.

## Full Phase Ledger

### Phases 0-49

- [x] Phase 0: `manifest`
- [x] Phase 1: `bootstrap`
- [x] Phase 2: `core native`
- [x] Phase 3: `dtype constants`
- [x] Phase 4: `array bootstrap`
- [x] Phase 5: `core ops`
- [x] Phase 6: `creation ops`
- [x] Phase 7: `index concat stack split`
- [x] Phase 8: `unary logical ops`
- [x] Phase 9: `linalg broadcast`
- [x] Phase 10: `bool compare select`
- [x] Phase 11: `compare close`
- [x] Phase 12: `nan inf predicates`
- [x] Phase 13: `elementwise clip`
- [x] Phase 14: `boolean reductions`
- [x] Phase 15: `axis manipulation`
- [x] Phase 16: `repeat tile`
- [x] Phase 17: `meshgrid`
- [x] Phase 18: `inner outer`
- [x] Phase 19: `logical ops`
- [x] Phase 20: `bitwise ops`
- [x] Phase 21: `sort argsort`
- [x] Phase 22: `min max arg reductions`
- [x] Phase 23: `prod cumulative`
- [x] Phase 24: `var std`
- [x] Phase 25: `median`
- [x] Phase 26: `arithmetic function ops`
- [x] Phase 27: `unary math extended`
- [x] Phase 28: `inverse trig conversion`
- [x] Phase 29: `log round div shift`
- [x] Phase 30: `shape manipulation`
- [x] Phase 31: `construct like tri`
- [x] Phase 32: `diag trace`
- [x] Phase 33: `topk partition`
- [x] Phase 34: `roll`
- [x] Phase 35: `take ops`
- [x] Phase 36: `nan to num`
- [x] Phase 37: `softmax logsumexp`
- [x] Phase 38: `kron`
- [x] Phase 39: `view like ops`
- [x] Phase 40: `shape alias ops`
- [x] Phase 41: `special math cumext`
- [x] Phase 42: `index update pad`
- [x] Phase 43: `tensordot einsum`
- [x] Phase 44: `gemm helpers`
- [x] Phase 45: `quantization basics`
- [x] Phase 46: `quantized gather`
- [x] Phase 47: `convolution basics`
- [x] Phase 48: `convolution advanced`
- [x] Phase 49: `depends and npy io`

### Phases 50-99

- [x] Phase 50: `io formats`
- [x] Phase 51: `random core`
- [x] Phase 52: `random extended`
- [x] Phase 53: `fft complex`
- [x] Phase 54: `fft real shift`
- [x] Phase 55: `linalg decompositions`
- [x] Phase 56: `linalg solve inverse`
- [x] Phase 57: `linalg factor eigen`
- [x] Phase 58: `runtime eval compile`
- [x] Phase 59: `fast ops`
- [x] Phase 60: `transforms autodiff`
- [x] Phase 61: `export graph`
- [x] Phase 62: `metal surface`
- [x] Phase 63: `distributed surface`
- [x] Phase 64: `parity surface`
- [x] Phase 65: `io npz load`
- [x] Phase 66: `io savez`
- [x] Phase 67: `stream fft`
- [x] Phase 68: `stream linalg fast dist`
- [x] Phase 69: `metal kernel`
- [x] Phase 70: `cuda kernel surface`
- [x] Phase 71: `precompiled cuda kernel`
- [x] Phase 72: `transforms argnames tree`
- [x] Phase 73: `compile checkpoint tree kwargs`
- [x] Phase 74: `export kwargs roundtrip`
- [x] Phase 75: `custom function surface`
- [x] Phase 76: `export kwargs only`
- [x] Phase 77: `api inventory contract`
- [x] Phase 78: `core signature inventory`
- [x] Phase 79: `array instance surface`
- [x] Phase 80: `module presence`
- [x] Phase 81: `build stability contract`
- [x] Phase 82: `parity report artifact`
- [x] Phase 83: `device type surface`
- [x] Phase 84: `stream context surface`
- [x] Phase 85: `device equality compat`
- [x] Phase 86: `array protocol compat`
- [x] Phase 87: `array dunder arithmetic`
- [x] Phase 88: `array getsetitem`
- [x] Phase 89: `array state protocol`
- [x] Phase 90: `array dlpack surface`
- [x] Phase 91: `custom function grad integration`
- [x] Phase 92: `custom function jvp vjp integration`
- [x] Phase 93: `custom function vmap integration`
- [x] Phase 94: `grad arg validation`
- [x] Phase 95: `compile constant validation`
- [x] Phase 96: `utils tree map`
- [x] Phase 97: `utils tree map with path`
- [x] Phase 98: `array remaining surface`
- [x] Phase 99: `array property surface`

### Phases 100-148

- [x] Phase 100: `array name parity contract`
- [x] Phase 101: `parity report checks`
- [x] Phase 102: `extension module presence`
- [x] Phase 103: `module name parity contract`
- [x] Phase 104: `parity report module checks`
- [x] Phase 105: `core class surface parity`
- [x] Phase 106: `package inventory contract`
- [x] Phase 107: `package report contract`
- [x] Phase 108: `package skeleton`
- [x] Phase 109: `package parity`
- [x] Phase 110: `package parity`
- [x] Phase 111: `package parity`
- [x] Phase 112: `package parity`
- [x] Phase 113: `package parity`
- [x] Phase 114: `package parity`
- [x] Phase 115: `package parity`
- [x] Phase 116: `package parity`
- [x] Phase 117: `package parity`
- [x] Phase 118: `package parity`
- [x] Phase 119: `package parity`
- [x] Phase 120: `package parity`
- [x] Phase 121: `package parity`
- [x] Phase 122: `package parity`
- [x] Phase 123: `package parity`
- [x] Phase 124: `package parity`
- [x] Phase 125: `package parity`
- [x] Phase 126: `package parity`
- [x] Phase 127: `package parity`
- [x] Phase 128: `package parity`
- [x] Phase 129: `package parity`
- [x] Phase 130: `package parity`
- [x] Phase 131: `package parity`
- [x] Phase 132: `package parity`
- [x] Phase 133: `package parity`
- [x] Phase 134: `package parity`
- [x] Phase 135: `package parity`
- [x] Phase 136: `package parity`
- [x] Phase 137: `package parity`
- [x] Phase 138: `package parity`
- [x] Phase 139: `package parity`
- [x] Phase 140: `package parity`
- [x] Phase 141: `package parity`
- [x] Phase 142: `package parity`
- [x] Phase 143: `package parity`
- [x] Phase 144: `package parity`
- [x] Phase 145: `package parity`
- [x] Phase 146: `package parity`
- [x] Phase 147: `package parity`
- [x] Phase 148: `package parity`

### Phases 149-210

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
- [x] Phase 188: `Embedding` + container parity
- [x] Phase 189: dropout parity
- [x] Phase 190: activation function/class parity
- [x] Phase 191: positional encoding parity
- [x] Phase 192: quantized linear/embedding scaffolding parity
- [x] Phase 193: convolution layers parity
- [x] Phase 194: transposed convolution parity
- [x] Phase 195: pooling parity
- [x] Phase 196: normalization parity
- [x] Phase 197: recurrent layers parity
- [x] Phase 198: transformer layers parity
- [x] Phase 199: upsample parity
- [x] Phase 200: distributed layer helpers parity
- [x] Phase 201: distributed linear classes parity
- [x] Phase 202: quantized distributed classes parity
- [x] Phase 203: distributed utils common parity
- [x] Phase 204: distributed config parity
- [x] Phase 205: distributed launch parity
- [x] Phase 206: parity harness for optimizer/loss/module golden cases
- [x] Phase 207: parity harness for layer golden cases
- [x] Phase 208: parity harness for distributed-utils golden cases
- [x] Phase 209: consolidated parity report with uncovered behavior list
- [x] Phase 210: final cleanup and strict full-suite green gate

### Phases 211-219

- [x] Phase 211: DLPack runtime parity (`__dlpack__` export object + Ruby roundtrip contract)
- [x] Phase 212: distributed launch process runtime parity (`RemoteProcess` spawn/terminate/exit semantics)
- [x] Phase 213: launch backend assembly parity (`launch_ring`, `launch_nccl`, `launch_jaccl`)
- [x] Phase 214: MPI launch parity (`get_mpi_libname`, `launch_mpi` hostfile/env assembly)
- [x] Phase 215: launch CLI parity (`main` argument parsing, backend dispatch, script resolution)
- [x] Phase 216: distributed config command parity (`add_ips`, `check_rdma`, `check_ssh_connections`, `IPConfigurator`)
- [x] Phase 217: distributed hostfile configuration parity (`configure_ring`, `configure_jaccl`, `configure_jaccl_ring`, `prepare_*`)
- [x] Phase 218: distributed config CLI parity (`main`, backend/over selection, output behavior)
- [x] Phase 219: consolidated distributed+dlpack functional harness + strict full-suite gate

### Phases 220-252 (Python Test Gap Closure)

- [x] Phase 220: lock Python-vs-Ruby gap baseline artifact (method-level diff + module-level coverage report in `ruby/parity/`)
- [x] Phase 221: array numpy conversion parity (`test_array_np_conversion`, `test_array_np_dtype_conversion`, `test_add_numpy`) via Ruby-driven Python interop harness tests
- [x] Phase 222: array non-contiguous import/copy semantics parity (`test_array_from_noncontiguous_np`, `test_np_array_conversion_copies_by_default`)
- [x] Phase 223: array serialization/state parity for pickle-like workflows (`test_array_pickle`, `test_load_from_pickled_np`)
- [x] Phase 224: array buffer protocol parity and lifetime checks (`test_buffer_protocol`, `test_buffer_protocol_ref_counting`, `test_array_view_ref_counting`, `test_buffer_protocol_tf`)
- [x] Phase 225: array indexing update edge parity (`test_setitem_with_boolean_mask`, list-based get/set index combinations)
- [x] Phase 226: array graph/index stress parity (`test_deep_graphs`, `test_siblings_without_eval`, `test_large_indices`)
- [x] Phase 227: autograd scatter overwrite parity (`test_scatter_vjp`)
- [x] Phase 228: autograd scatter max/min tie behavior parity (`test_scatter_max_vjp`, `test_scatter_min_vjp`)
- [x] Phase 229: autograd indexing gradient parity (`test_put_along_axis_grads`, `test_slice_grads`, `test_topk_grad`)
- [x] Phase 230: autograd advanced structure/type parity (`test_complex_vjps`, `test_flatten_unflatten_vjps`, `test_concatenate_vjps`, `test_matmul_jvps`)
- [x] Phase 231: autograd stability parity (`test_reduce_jvp`, `test_cumprod_grad`, `test_grad_ids_pre_post`, `test_grad_with_inplace_update`, `test_leaks`)
- [x] Phase 232: compile shapeless baseline parity (`test_shapeless_compile`, `test_shapeless_compile_with_broadcasts`, `test_shapeless_compile_with_reduction`)
- [x] Phase 233: compile shapeless op coverage parity (`test_shapeless_compile_unflatten`, `test_shapeless_compile_gather`, `test_shapeless_compile_full_like`, `test_shapeless_compile_matmul`, `test_shapeless_compile_slice_update`, `test_shapeless_compile_with_reshape`)
- [x] Phase 234: compile dynamic/many-arity parity (`test_compile_dynamic_dims`, `test_compile_many_inputs`, `test_compile_many_outputs`)
- [x] Phase 235: compile type/constant validation parity (`test_unsupported_input_types`, `test_compile_with_none`, `test_compile_types`, dtype-heavy compile checks)
- [x] Phase 236: compile changing-structure parity (`test_compile_changing_outputs`, `test_compile_changing_outputs_with_state`, `test_outputs_changing`, `test_compile_output_with_siblings`)
- [x] Phase 237: compile runtime stability parity (`test_compiled_preserves_attributes`, `test_compile_donates_input_buffer`, `test_leaks`, shared-broadcast and wrapped-compiled checks)
- [x] Phase 238: quantized matvec scenario parity via `quantized_matmul`/`qqmm` (`qmv`/`qvm`/`qqmv` behavior coverage)
- [x] Phase 239: quantized mode/shape error parity (`test_mode_error_cases`, `test_non_multiples`, small-non-multiple vector/matrix paths)
- [x] Phase 240: quantized gradient parity I (`test_qmm_vjp`, `test_qmm_jvp`, `test_gather_matmul_grad`)
- [x] Phase 241: quantized gradient parity II (`test_gather_qmm_grad`, `test_vjp_scales_biases`, `test_fp_vjp_scales_throws`, `test_quantize_strided`)
- [x] Phase 242: fast SDPA masking/layout parity (`test_sdpa_broadcast_mask`, `test_sdpa_promote_mask`, `test_sdpa_noncontiguous_inputs`, `test_sdpa_attention_sinks`)
- [x] Phase 243: fast SDPA numeric edge parity (`test_sdpa_fully_masked`, `test_sdpa_inf_score`, `test_sdpa_nan_bug`, vector/few-query variants)
- [x] Phase 244: fast SDPA autodiff parity (`test_sdpa_grad`, `test_vjp`, `test_grad`, `test_sdpa_sliced`)
- [x] Phase 245: vmap foundational behavioral parity (`test_unary`, `test_binary`, `test_tree`, `test_vmap_indexing`, `test_vmap_reduce`, `test_vmap_argreduce`, `test_vmap_mean`, `test_mismatch_input_sizes`)
- [x] Phase 246: vmap linear algebra and gather/scatter parity (`test_vmap_matmul`, `test_vmap_svd`, `test_vmap_inverse`, `test_vmap_gather`, `test_vmap_scatter`, `test_vmap_take_along_axis`, `test_vmap_put_along_axis`)
- [x] Phase 247: vmap remaining stability/type parity (`test_vmap_concatenate`, `test_vmap_split_vmap`, `test_vmap_masked_scatter`, `test_vmap_flatten`, `test_vmap_conv`, `test_vmap_types`, `test_leaks`)
- [x] Phase 248: memory control and accounting parity (`test_memory_info`, `test_wired_memory`, `test_active_memory_count`)
- [x] Phase 249: graph export parity for file-like outputs (`test_to_dot` stream/file-object behavior in addition to path-based output)
- [x] Phase 250: load/save format edge parity (`test_load_npy_dtype`, GGUF metadata variants, FP8 load paths, non-contiguous load, load donation behavior)
- [x] Phase 251: export/import advanced parity (`test_export_variable_inputs`, shapeless export/import, control-flow export, callback/custom-kernel export, constants/multi-export/leak paths)
- [x] Phase 252: final strict parity gate (all new Ruby phases green, full Ruby suite green, refreshed gap artifact shows zero unresolved targeted gaps)

### Phases 253-254 (Cross-Device Execution Gate)

- [x] Phase 253: add cross-device runtime tests for `DEVICE` selection and GPU-safe nested array construction (`phase253_device_matrix_runtime_test.rb`)
- [x] Phase 254: enforce full-suite dual-run gate with current tree green on both `DEVICE=cpu` and `DEVICE=gpu`

## Phase 220+ Execution Notes

For each phase above, execute in strict red/green order and do not advance early:
1. Add failing Ruby phase tests that mirror named Python test behaviors.
2. Implement minimum Ruby wrapper/native changes to satisfy only that phase.
3. Run the new phase tests, then full Ruby test suite.
4. Update parity artifact and mark the phase complete only when all gates pass.
