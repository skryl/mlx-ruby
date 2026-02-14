# Ruby DSL Phased Implementation Plan

## Goal

Add a Ruby-native DSL for MLX bindings under `lib/mlx/dsl` that preserves compatibility with existing `MLX::Core`, `MLX::NN`, and `MLX::Optimizers` behavior while improving ergonomics for model definition and training.

This plan is intentionally separate from parity-report artifacts. The DSL is a Ruby-only extension and should not be counted in Python package-parity file checks.

## Design Principles

1. Keep existing APIs stable.
2. Build DSL features as sugar over existing primitives.
3. Keep all DSL state interoperable with `MLX::NN::Module#parameters`, `#trainable_parameters`, `#update`, and optimizer flows.
4. Implement with red/green testing:
   - Red: add failing tests for new behavior.
   - Green: implement minimal code to pass.
   - Refactor: tighten internals without changing behavior.

## Phase 1: Foundation (`lib/mlx/dsl`)

### Deliverables

1. Add:
   - `lib/mlx/dsl.rb`
   - `lib/mlx/dsl/model.rb`
   - `lib/mlx/dsl/model_mixin.rb`
   - `lib/mlx/dsl/builder.rb`
   - `lib/mlx/dsl/train_step.rb`
2. Load DSL from `lib/mlx.rb`.
3. Add initial tests in `test/dsl/dsl_test.rb`.

### Ergonomics Target

```ruby
class Classifier < MLX::DSL::Model
  layer :net do
    sequential do
      linear 784, 256
      relu
      dropout 0.1
      linear 256, 10
    end
  end

  def call(x) = net.call(x)
end
```

## Phase 2: Model Declaration DSL

### Deliverables

1. `option` class macro with required/default behavior.
2. `layer` and `network` macros for submodule declaration.
3. `param` and `buffer` macros for array declaration and initialization.
4. Ensure declarations write through `self.<name>=` so module state tracking remains consistent.

### Ergonomics Target

```ruby
class Affine < MLX::DSL::Model
  option :in_dim
  option :out_dim
  option :use_bias, default: true

  param :weight, shape: -> { [out_dim, in_dim] }, init: ->(shape) { MLX::Core.normal(shape, 0.0, 0.02) }
  buffer :scale, shape: -> { [out_dim] }, init: ->(shape) { MLX::Core.ones(shape, MLX::Core.float32) }

  def call(x)
    y = MLX::Core.matmul(x, weight.T)
    y = MLX::Core.add(y, bias) if use_bias
    MLX::Core.multiply(y, scale)
  end
end
```

## Phase 3: Mixin Support for Existing Modules

### Deliverables

1. `MLX::DSL::ModelMixin` for classes that already inherit `MLX::NN::Module`.
2. Shared declaration behavior between `Model` and `ModelMixin`.

### Ergonomics Target

```ruby
class ResidualBlock < MLX::NN::Module
  include MLX::DSL::ModelMixin

  option :dims, default: 256

  layer(:proj) { linear dims, dims, bias: false }
  layer(:norm) { layer_norm dims }

  def call(x)
    MLX::Core.add(x, norm.call(proj.call(x)))
  end
end
```

## Phase 4: Builder/Composition Ergonomics

### Deliverables

1. Builder methods for common layers:
   - `linear`, `relu`, `dropout`, `conv2d`, `layer_norm`, etc.
2. `sequential` block collector.
3. Baseline branch-friendly structure for later graph helpers.

### Ergonomics Target

```ruby
layer :encoder do
  sequential do
    linear 512, 512
    relu
    dropout 0.1
    linear 512, 256
  end
end
```

## Phase 5: Training Ergonomics

### Deliverables

1. `train_step` helper that wraps:
   - `MLX::NN.value_and_grad`
   - optional `MLX::Optimizers.clip_grad_norm`
   - `optimizer.update`
2. Hook support for training lifecycle events:
   - `:before_step`
   - `:after_backward`
   - `:after_step`
3. Mode helpers:
   - `train_mode { ... }`
   - `eval_mode { ... }`

### Ergonomics Target

```ruby
step = model.train_step(optimizer: optimizer, clip_grad_norm: 1.0) do |x:, y:|
  logits = model.call(x)
  MLX::NN.cross_entropy(logits, y, reduction: "mean")
end

step.on(:after_step) do |ctx|
  puts "step=#{ctx[:step]} loss=#{ctx[:loss].item}"
end

loss = step.call(x: batch_x, y: batch_y)
```

## Phase 6: Optimizer Group Ergonomics

### Deliverables

1. `optimizer_groups` builder on model instances.
2. `group(matcher)` declarations producing `MLX::Optimizers::MultiOptimizer`.
3. Path matcher support (`Regexp`, `String`, `Proc`).

### Ergonomics Target

```ruby
opt = model.optimizer_groups do
  group(/^encoder\./) { MLX::Optimizers::AdamW.new(learning_rate: 1e-4) }
  group(nil)          { MLX::Optimizers::SGD.new(learning_rate: 5e-3) }
end
```

## Phase 7: Parameter Selection Helpers

### Deliverables

1. `freeze_paths!(matcher)` and `unfreeze_paths!(matcher)` helpers.
2. Match by full flattened parameter path.

### Ergonomics Target

```ruby
model.freeze_paths!(/^encoder\./)
model.unfreeze_paths!(/^head\./)
```

## Phase 8: Parity Tooling Compatibility

### Deliverables

1. Update `tools/parity/generate_package_inventory.rb` ignore list so:
   - `lib/mlx/dsl.rb`
   - `lib/mlx/dsl/**/*.rb`
   are excluded from package file parity diffs.
2. Keep existing parity checks unchanged for Python-equivalent surfaces.

## Phase 9: Documentation and Examples

### Deliverables

1. Add DSL section to `README.md`.
2. Provide concise end-to-end examples:
   - MLP classification model
   - Mix-in based module
   - `train_step` with hooks
   - optimizer groups

## Phase 10: V2 Extensions (Post-V1)

### Deliverables

1. Graph helpers:
   - `residual`
   - `branch`
   - `concat`
2. Checkpoint helpers for model + optimizer state.
3. Lightweight trainer wrapper.

## Phase 11: Trainer UX Enhancements

### Deliverables

1. Hook shorthand methods (in addition to `on`):
   - `TrainStep`: `before_step`, `after_backward`, `after_step`
   - `Trainer`: `before_fit`, `before_epoch`, `after_batch`, `after_epoch`, `checkpoint`, `after_fit`
2. Custom monitor metrics for fit reports:
   - `monitor:` label in report output
   - `metric:` callable receiving epoch context
3. Checkpoint path templating:
   - `%{epoch}`, `%{monitor}`, `%{monitor_name}`, `%{epoch_loss}`, `%{improved}`
4. Early stopping controls:
   - `patience:`
   - `min_delta:`
   - report fields `epochs_ran` and `stopped_early`

### Ergonomics Target

```ruby
trainer = model.trainer(optimizer: optimizer) do |x:, y:|
  logits = model.call(x)
  MLX::NN.cross_entropy(logits, y, reduction: "mean")
end

trainer.before_epoch { |ctx| puts "epoch=#{ctx[:epoch]}" }
trainer.after_batch { |ctx| puts "batch=#{ctx[:batch_index]} loss=#{ctx[:loss_value]}" }

report = trainer.fit_report(
  dataset,
  epochs: 20,
  monitor: :peak_loss,
  monitor_mode: :max,
  metric: ->(ctx) { ctx.fetch(:epoch_losses).max },
  checkpoint_path: "ckpts/ep-%{epoch}-m-%{monitor}.bin",
  save_best: true,
  patience: 2,
  min_delta: 1e-4
)
```

## Phase 12: Data Ergonomics

### Deliverables

1. Streaming-friendly training datasets:
   - accept `Enumerable` without forcing `to_a`
   - accept dataset factories (`Proc`) that return per-epoch enumerables
2. Optional validation dataset loop:
   - `validation_data:` and `validation_reduce:`
   - include `val_loss` and `validation_batches` in epoch reports
3. Native monitoring for validation:
   - `monitor: :val_loss` without requiring a custom metric proc

### Ergonomics Target

```ruby
train_data = ->(epoch:) { shuffled_batches_for(epoch) }
val_data = ->(epoch:) { heldout_batches_for(epoch) }

report = trainer.fit_report(
  train_data,
  epochs: 10,
  reduce: :mean,
  validation_data: val_data,
  validation_reduce: :mean,
  monitor: :val_loss,
  monitor_mode: :min,
  save_best: true,
  checkpoint_path: "checkpoints/epoch-%{epoch}-val-%{monitor}.bin"
)
```

## Phase 13: Enumerable and Batch Pipeline Ergonomics

### Deliverables

1. Safer multi-epoch behavior for single-pass datasets:
   - `strict_data_reuse:` option to detect exhausted non-rewindable datasets across epochs
   - clear error pointing users to dataset factories
2. Batch transforms/collation hooks:
   - `train_transform:`
   - `validation_transform:`
   - supports transforms that receive `batch`, `epoch`, `batch_index`, `kind`, and `trainer`
3. Hardened dataset factory signatures:
   - support positional, keyword, and mixed signatures (e.g. `->(epoch, kind:)`)
   - explicit errors when required parameters are unsupported
4. Optional per-batch loss retention:
   - `keep_losses:` control for long-running jobs
   - preserves epoch-level reporting while avoiding unbounded `losses` arrays

### Ergonomics Target

```ruby
train_data = ->(epoch, kind:) { stream_train_batches(epoch, kind: kind) }
val_data = ->(epoch:) { stream_validation_batches(epoch) }

report = trainer.fit_report(
  train_data,
  epochs: 20,
  strict_data_reuse: true,
  train_transform: ->(batch, epoch:, batch_index:) { collate_train(batch, epoch, batch_index) },
  validation_data: val_data,
  validation_transform: ->(batch, epoch:) { collate_val(batch, epoch) },
  monitor: :val_loss,
  keep_losses: false
)
```

## Phase 14: Validation Lifecycle Hooks

### Deliverables

1. Add validation hook events on `MLX::DSL::Trainer`:
   - `before_validation`
   - `after_validation_batch`
   - `after_validation`
2. Expose hook shorthand methods matching the above events.
3. Include validation hook context fields:
   - `epoch`
   - `batch_index` (for per-batch hook)
   - `loss` and `loss_value`
   - reduced `val_loss` for `after_validation`

### Ergonomics Target

```ruby
trainer.before_validation { |ctx| puts "epoch=#{ctx[:epoch]} val:start" }
trainer.after_validation_batch { |ctx| puts "val_batch=#{ctx[:batch_index]} loss=#{ctx[:loss_value]}" }
trainer.after_validation { |ctx| puts "val_loss=#{ctx[:val_loss]}" }
```

## Phase 15: Native Integration Coverage For Data Ergonomics

### Deliverables

1. Add integration tests to `test/dsl/dsl_test.rb` for:
   - `train_transform`
   - `validation_transform`
   - `strict_data_reuse`
   - `keep_losses: false`
2. Ensure new assertions run against real `MLX::Core::Array` and optimizer flows.
3. Keep tests deterministic and fast (small toy datasets).

## Phase 16: Runnable DSL Examples

### Deliverables

1. Add `examples/dsl/` scripts for:
   - streaming per-epoch dataset factory
   - validation monitoring (`monitor: :val_loss`)
   - long-running memory-friendly reporting (`keep_losses: false`)
2. Keep examples executable via `bundle exec ruby examples/dsl/<name>.rb`.
3. Reference examples from `README.md`.

## Phase 17: No-Native Load Resilience

### Deliverables

1. Remove eager native-dependent defaults in DSL class macros where possible.
2. Ensure requiring DSL files does not raise when native extension is unavailable.
3. Prefer lazy dtype/default resolution at runtime instead of class definition time.

## Phase 18: Test Helper Rebuild Policy

### Deliverables

1. Reduce unnecessary forced rebuilds in `test/test_helper.rb`.
2. When a loadable native bundle already exists, avoid rebuild attempts that require unavailable source trees.
3. Preserve explicit rebuild behavior when `MLX_RUBY_FORCE_REBUILD=1`.

## Red/Green Execution Plan

1. Add tests for each DSL behavior in `test/dsl/dsl_test.rb` (red).
2. Implement minimum code in `lib/mlx/dsl/**` to pass tests (green).
3. Refactor and tighten internals while maintaining passing tests.
4. Run targeted suite first, then broader suite.

## Ergonomics Execution Track (Phases 19+)

This track focuses on reducing friction when building and running models with Ruby-native DSL behavior and dynamic data pipelines.

### Phase 19: DSL Declaration Safety and Error Quality

#### Problem

The DSL currently accepts some invalid declarations and surfaces unclear initialization errors for unknown options.

#### Deliverables

1. Reject non-module results from `layer` / `network` declarations.
2. Raise explicit unknown option errors for declared DSL options instead of falling through to generic Ruby argument errors.
3. Keep compatibility for classes that intentionally handle extra kwargs in their own initializer.

#### Red (tests first)

1. Add failing test for unknown DSL option handling in `test/dsl/dsl_test.rb`.
2. Add failing test proving layer declarations must materialize `MLX::NN::Module`.

#### Green (minimum implementation)

1. Tighten option extraction / initializer validation in `lib/mlx/dsl/model_mixin.rb`.
2. Validate layer declaration materialization type in `lib/mlx/dsl/model_mixin.rb`.

#### Exit Criteria

1. New tests pass.
2. Existing DSL declaration tests remain green.

### Phase 20: Dataset Factory Invocation Reliability

#### Problem

Dataset factories with zero-arity blocks are mis-invoked, and non-rewind errors can be mislabeled as rewind failures.

#### Deliverables

1. Correct dataset factory invocation for:
   - `-> { ... }`
   - `->(epoch) { ... }`
   - `->(epoch:) { ... }`
   - mixed signatures
2. Limit rewind-specific error wrapping to rewind failures only.

#### Red (tests first)

1. Add failing trainer unit test for zero-arity dataset factory support.

#### Green (minimum implementation)

1. Fix factory invocation branch in `lib/mlx/dsl/trainer.rb`.
2. Narrow error handling in dataset/rewind path in `lib/mlx/dsl/trainer.rb`.

#### Exit Criteria

1. New trainer tests pass.
2. Existing dataset factory and strict-data-reuse tests stay green.

### Phase 21: Variadic Composition Through Sequential

#### Problem

`MLX::NN::Sequential` currently forces a single positional input, reducing composability for multi-arg modules and Ruby-style dynamic wiring.

#### Deliverables

1. Support first-layer invocation with `*args, **kwargs`.
2. Support forwarding intermediary payloads as:
   - positional args when payload is `Array`
   - kwargs when payload is `Hash`
   - single positional argument otherwise
3. Preserve existing single-tensor flow behavior.

#### Red (tests first)

1. Add failing unit test in `test/dsl/dsl_graph_unit_test.rb` covering multi-arg + kwargs + array forwarding across sequential layers.

#### Green (minimum implementation)

1. Update `MLX::NN::Sequential#call` in `lib/mlx/nn/layers/containers.rb`.

#### Exit Criteria

1. New composition test passes.
2. Existing graph/builder tests remain green.

### Phase 22: Compile-Aware Training Step

#### Deliverables

1. Add optional `compile:` mode for `train_step` and trainer internals.
2. Add explicit sync policy controls (`:none`, `:step`, `:epoch`) to avoid implicit runtime behavior.
3. Add focused parity/integration tests for compile + hooks + checkpoint interactions.

### Phase 23: Checkpoint Format Interoperability

#### Deliverables

1. Introduce native checkpoint formats (`.npz` / `.safetensors`) for model parameters.
2. Preserve metadata and optional optimizer state with versioned schema.
3. Keep current marshal path as compatibility fallback during migration window.

### Phase 24: Builder Surface + Batch Collation Ergonomics

#### Deliverables

1. Expand DSL builder coverage to match practical `MLX::NN` usage (recurrent/transformer/transpose-conv paths).
2. Add first-class batch collation helpers for common train/eval datasets.
3. Add end-to-end examples demonstrating lower boilerplate and clearer training loops.

### Phase 25: Docs + Example Refresh For New Ergonomics

#### Deliverables

1. Update README DSL section with:
   - `compile:` / `sync:` usage on `train_step` and `trainer`
   - `collate:` / `validation_collate:` examples
   - native checkpoint usage (`.npz` / `.safetensors`) and metadata behavior
2. Add runnable example scripts for:
   - built-in collate schemas and mapping-based collation
   - compile/sync controls and native checkpoint roundtrip
3. Keep examples executable via `bundle exec ruby examples/dsl/<name>.rb`.

### Phase 26: Context-Aware Collation Callables

#### Problem

Collation Procs are currently batch-only, which limits Ruby-style dynamic shaping based on epoch and trainer runtime context.

#### Deliverables

1. Allow `collate:` and `validation_collate:` callables to accept dynamic signatures:
   - positional batch input
   - optional keyword context (`epoch`, `batch_index`, `kind`, `trainer`)
2. Allow mapping-collate `Proc` selectors to use the same dynamic signature support.
3. Preserve existing simple `->(batch) { ... }` behavior.

#### Red (tests first)

1. Add failing trainer unit test for context-aware `collate` callable signatures.
2. Add failing trainer unit test for context-aware mapping selector Procs.
3. Add failing trainer unit test for context-aware `validation_collate`.

#### Green (minimum implementation)

1. Thread epoch/batch context through trainer collation call sites.
2. Add signature-aware callable invocation helper in `lib/mlx/dsl/trainer.rb`.
3. Route mapping selector Procs through the same callable helper.

#### Exit Criteria

1. New context-aware collation tests pass.
2. Existing collation + transform tests remain green.

### Phase 27: Keyword-Normalized Hash Batch Dispatch

#### Problem

Ruby keyword lambdas (for `train_step` and validation loss blocks) fail when dataset hashes use string keys, even when keys are semantically correct.

#### Deliverables

1. Normalize top-level hash keys to symbols before keyword dispatch in train and validation execution paths.
2. Raise clear errors when normalization causes duplicate keyword collisions (for example `"x"` and `:x` in the same batch).
3. Keep array and scalar batch dispatch behavior unchanged.

#### Red (tests first)

1. Add failing trainer unit test proving train batches with string keys are accepted for keyword step call signatures.
2. Add failing trainer unit test proving validation batches with string keys are accepted for keyword loss signatures.
3. Add failing trainer unit test asserting duplicate normalized keys raise explicit error.

#### Green (minimum implementation)

1. Normalize hash keys in `__dsl_run_batch` and `__dsl_run_validation_batch`.
2. Add duplicate-key detection and error reporting helper in `lib/mlx/dsl/trainer.rb`.

#### Exit Criteria

1. New key-normalization tests pass.
2. Existing trainer behavior stays green across full DSL test suite.

### Phase 28: Inline Callable Layer Ergonomics

#### Problem

Users currently need ad-hoc module classes for simple one-off tensor transforms in builder graphs, which adds boilerplate and slows experimentation.

#### Deliverables

1. Add a dedicated DSL callable-layer wrapper module for inline Ruby lambdas/procs.
2. Add builder helpers for inline callable layers:
   - `fn(callable = nil, &block)`
   - `lambda_layer` alias
3. Preserve full variadic call forwarding (`*args, **kwargs`) through callable layers.

#### Red (tests first)

1. Add failing graph unit test for `Builder#fn` returning a DSL module wrapper.
2. Add failing graph unit test for variadic arg/kwarg forwarding through callable layers.
3. Add failing graph unit test that missing callable/block raises a clear argument error.

#### Green (minimum implementation)

1. Add `MLX::DSL::Callable` module in `lib/mlx/dsl/graph_modules.rb`.
2. Add `fn` and `lambda_layer` builder methods in `lib/mlx/dsl/builder.rb`.

#### Exit Criteria

1. New callable-layer tests pass.
2. Existing graph/builder/trainer DSL tests remain green.

### Phase 29: Composition Input Normalization

#### Problem

Composition helpers (`sequential`, `branch`, `concat`, `sum`, `residual`) currently accept heterogeneous inputs without normalization, which can leak raw classes/callables into layer stacks and break module tracking.

#### Deliverables

1. Normalize composition entries to `MLX::NN::Module` instances:
   - instantiate module classes
   - wrap callables with `MLX::DSL::Callable`
2. Raise clear `TypeError` for invalid composition entries at build time.
3. Preserve existing module-instance behavior.

#### Red (tests first)

1. Add failing graph unit tests proving `sequential` and `branch` normalize class + callable entries into module instances.
2. Add failing graph unit test proving invalid composition entries are rejected early.

#### Green (minimum implementation)

1. Add module-entry normalization path in `lib/mlx/dsl/builder.rb`.
2. Update module collection fallback to include non-nil block returns so normalization applies consistently.

#### Exit Criteria

1. New normalization tests pass.
2. Existing graph and trainer DSL suites remain green.

### Phase 30: Checkpoint Path Directory Ergonomics

#### Problem

Saving checkpoints to nested output paths currently fails when parent directories do not already exist, adding avoidable filesystem setup boilerplate.

#### Deliverables

1. Automatically create parent directories for marshal checkpoint writes.
2. Automatically create parent directories for native checkpoint writes (`.npz` / `.safetensors`).
3. Preserve existing checkpoint payload/schema behavior.

#### Red (tests first)

1. Add failing DSL integration test proving marshal checkpoint save succeeds for non-existent nested directories.
2. Add failing DSL integration test proving native checkpoint save succeeds for non-existent nested directories.

#### Green (minimum implementation)

1. Add checkpoint parent-directory helper in `lib/mlx/dsl/model_mixin.rb`.
2. Invoke helper from both marshal and native save paths.

#### Exit Criteria

1. New checkpoint directory tests pass.
2. Existing checkpoint and trainer tests remain green.

### Phase 31: Polymorphic `Builder#layer` Input Ergonomics

#### Problem

`Builder#layer` currently assumes class-only inputs, which makes it inconsistent with the rest of the composition DSL and prevents direct reuse of module instances/callables.

#### Deliverables

1. Support `layer` inputs as:
   - `MLX::NN::Module` instance
   - `MLX::NN::Module` class (with constructor args/kwargs)
   - callable (`Proc`/`lambda`) wrapped as `MLX::DSL::Callable`
   - block form (`layer { |...| ... }`) as callable layer
2. Add clear errors for:
   - missing entry/block
   - ambiguous entry + block usage
   - constructor args passed to instance/callable entries
3. Keep existing class-constructor behavior intact.

#### Red (tests first)

1. Add failing graph unit tests covering instance/class/callable/block `layer` forms.
2. Add failing graph unit tests for invalid argument and missing-entry error paths.

#### Green (minimum implementation)

1. Expand `Builder#layer` dispatch logic in `lib/mlx/dsl/builder.rb`.
2. Add focused helper for constructor-argument validation on non-class entries.

#### Exit Criteria

1. New `layer` polymorphism tests pass.
2. Existing graph/builder/trainer DSL suites remain green.

### Phase 32: Declarative Layer Factory Arguments

#### Problem

Model declaration macros require extra lambda boilerplate when layer factories need constructor arguments derived from DSL options.

#### Deliverables

1. Extend `layer` / `network` declaration macros to accept factory constructor args/kwargs:
   - `layer :proj, MLX::NN::Linear, -> { dims }, -> { dims }, bias: false`
2. Resolve callable constructor args/kwargs in model context (same semantics as other DSL callables).
3. Keep clear rejection for ambiguous factory+block declarations.

#### Red (tests first)

1. Add failing DSL integration test for class factory with dynamic args/kwargs.
2. Add failing DSL integration test for callable factory with dynamic args/kwargs.
3. Add failing DSL integration test for factory+block ambiguity.

#### Green (minimum implementation)

1. Expand `ClassMethods#layer` / `#network` signatures and stored declaration payload.
2. Update layer materialization path in `lib/mlx/dsl/model_mixin.rb` to apply resolved factory args/kwargs.

#### Exit Criteria

1. New declaration factory-arg tests pass.
2. Existing DSL suite remains green.

### Phase 33: Batch Failure Diagnostics With Epoch/Index Context

#### Problem

When train/validation batch execution raises, errors currently surface without trainer location context, making dynamic pipeline failures slow to debug.

#### Deliverables

1. Include `kind`, `epoch`, and `batch_index` context in train batch execution errors.
2. Include `kind`, `epoch`, and `batch_index` context in validation batch execution errors.
3. Preserve original exception class and original message details.

#### Red (tests first)

1. Add failing trainer unit test for train-step failure message context.
2. Add failing trainer unit test for validation loss failure message context.

#### Green (minimum implementation)

1. Thread batch location metadata into internal train/validation batch runner calls.
2. Add shared error re-raise helper in `lib/mlx/dsl/trainer.rb`.

#### Exit Criteria

1. New batch-diagnostic tests pass.
2. Existing DSL suite remains green.

### Phase 34: Validation Loop Limit Control

#### Problem

Trainer exposes `limit:` for train batches but not validation batches, forcing custom dataset wrappers for quick validation sampling.

#### Deliverables

1. Add `validation_limit:` option to `Trainer#fit` / `fit_report`.
2. Apply per-epoch validation loop cap without changing existing reducer/checkpoint semantics.
3. Preserve default behavior when `validation_limit` is omitted.

#### Red (tests first)

1. Add failing trainer unit test proving validation batches can be capped and metrics are computed from the capped set.

#### Green (minimum implementation)

1. Extend `fit` keyword signature with `validation_limit`.
2. Add break condition in validation epoch iteration path.

#### Exit Criteria

1. New validation-limit test passes.
2. Existing DSL suite remains green.

### Phase 35: Extensionless Native Checkpoint Load Autodetection

#### Problem

Native checkpoints saved with explicit `format:` and extensionless base names require callers to remember the generated extension when loading.

#### Deliverables

1. For `load_checkpoint(path, format: nil)` where `path` has no extension and does not exist:
   - autodetect `#{path}.npz`
   - fallback autodetect `#{path}.safetensors`
2. Preserve existing explicit `format:` behavior and marshal path semantics.
3. Keep native payload/metadata loading unchanged.

#### Red (tests first)

1. Add failing DSL integration test proving extensionless load path auto-detects an existing native `.npz` checkpoint.

#### Green (minimum implementation)

1. Add load-path resolution helper in `lib/mlx/dsl/model_mixin.rb`.
2. Route `load_checkpoint` through resolved path before format dispatch.

#### Exit Criteria

1. New autodetect integration test passes.
2. Existing DSL suite remains green.

### Phase 36: Resumeable Trainer Runs From Checkpoints

#### Problem

Long-running experiments need restart safety, but trainer state (`epoch`, `best_metric`, `stale_epochs`) is currently not restored into `fit` / `fit_report`.

#### Deliverables

1. Add `resume_from:` option to `Trainer#fit` and `fit_report`.
2. Restore checkpoint-backed trainer state:
   - next start epoch (`epoch + 1`)
   - `best_metric`
   - `stale_epochs`
   - monitor consistency via `monitor_name`
3. Persist resume-relevant metadata during checkpoint saves for future resumes.
4. Include resume fields in report payload for observability:
   - `resume_from`
   - `resumed_from_epoch`
   - `start_epoch`

#### Red (tests first)

1. Add failing trainer unit test for successful continuation from checkpoint metadata.
2. Add failing trainer unit test proving early-stopping stale counter is restored on resume.
3. Add failing trainer unit test rejecting monitor mismatch between requested monitor and checkpoint monitor metadata.

#### Green (minimum implementation)

1. Extend trainer fit signature and epoch loop to begin from resumed epoch.
2. Add resume-state loader helper with backward-compatible `load_checkpoint` keyword negotiation.
3. Add additional checkpoint metadata fields (`stale_epochs`, `best_metric`, `next_epoch`) in trainer checkpoint writes.

#### Exit Criteria

1. Resume tests pass.
2. Full DSL test suite remains green.

### Phase 37: Inline Resume Payload Ergonomics

#### Problem

`resume_from:` currently assumes filesystem-backed checkpoints; dynamic Ruby workflows often already hold parsed checkpoint payloads in memory.

#### Deliverables

1. Allow `resume_from:` to accept an inline checkpoint payload `Hash`.
2. Bypass model `load_checkpoint` when inline payload is provided.
3. Preserve existing path-based resume behavior unchanged.

#### Red (tests first)

1. Add failing trainer unit test proving `resume_from: { ... }` resumes epochs/metrics without invoking `load_checkpoint`.

#### Green (minimum implementation)

1. Branch resume source handling in `lib/mlx/dsl/trainer.rb` for hash payloads vs path inputs.
2. Keep report resume metadata consistent (`resume_from` nil for inline payloads, `resumed_from_epoch` retained).

#### Exit Criteria

1. Inline resume payload tests pass.
2. Full DSL suite remains green.

### Phase 38: Callable Resume Loader Support

#### Problem

Dynamic orchestration layers often need to compute resume state at runtime (for example, choose latest checkpoint per monitor), which path-only resume wiring does not express cleanly.

#### Deliverables

1. Allow `resume_from:` to accept a callable loader.
2. Support dynamic callable signatures for loader context:
   - `trainer`
   - `model`
   - `optimizer`
   - `monitor_name`
3. Allow loader return values as:
   - inline checkpoint payload hash
   - checkpoint path
   - `nil` (no resume)

#### Red (tests first)

1. Add failing trainer unit test proving callable loader receives context and can return an inline payload without invoking `load_checkpoint`.

#### Green (minimum implementation)

1. Add callable resume loader invocation helper in `lib/mlx/dsl/trainer.rb`.
2. Route resume-source normalization through callable/hash/path branches.

#### Exit Criteria

1. Callable resume loader tests pass.
2. Full DSL suite remains green.

### Phase 39: Resume Progress Telemetry In Fit Reports

#### Problem

`epochs_ran` alone is ambiguous in resumed runs because it reports only epochs executed in the current invocation.

#### Deliverables

1. Add explicit fit-report progress fields:
   - `epochs_target` (requested total epoch target)
   - `epochs_completed` (effective total progress including resumed offset)
2. Preserve existing `epochs_ran` semantics for backward compatibility.

#### Red (tests first)

1. Add failing trainer unit assertions for new progress fields in fresh runs.
2. Add failing trainer unit assertions for new progress fields in resumed runs.

#### Green (minimum implementation)

1. Track `total_epochs` in `Trainer#fit`.
2. Emit `epochs_target` / `epochs_completed` in report payload.

#### Exit Criteria

1. New report telemetry tests pass.
2. Full DSL suite remains green.

### Phase 40: Checkpoint Template `next_epoch` Placeholder

#### Problem

Checkpoint naming templates support `%{epoch}` but resume flows often need forward-looking names aligned with the next epoch index.

#### Deliverables

1. Add `%{next_epoch}` template token support to trainer checkpoint path rendering.
2. Keep existing checkpoint template tokens and error behavior unchanged.

#### Red (tests first)

1. Add failing trainer unit test proving `%{next_epoch}` renders as `epoch + 1`.

#### Green (minimum implementation)

1. Extend `__dsl_checkpoint_path` interpolation map with `next_epoch`.

#### Exit Criteria

1. New template test passes.
2. Full DSL suite remains green.

### Phase 41: First-Class Dataset Pipeline DSL

#### Problem

Training data preparation still relies on ad-hoc Ruby enumerator wiring, which makes common transforms repetitive and less composable.

#### Deliverables

1. Add a dataset pipeline wrapper under `MLX::DSL::Data` with chainable transforms:
   - `map`
   - `filter`
   - `batch`
   - `take`
   - `repeat`
2. Keep pipeline output compatible with existing trainer dataset expectations (`#each`, rewind/factory behavior).
3. Preserve lazy semantics by default to avoid eager memory blowups.

#### Red (tests first)

1. Add failing unit tests for transform chaining and stable iteration semantics across epochs.
2. Add failing trainer integration test proving pipeline output works with `fit`/`fit_report`.

#### Green (minimum implementation)

1. Add pipeline wrapper implementation in `lib/mlx/dsl/data_pipeline.rb`.
2. Load from `lib/mlx/dsl.rb`.
3. Integrate with trainer data paths without changing existing dataset APIs.

#### Exit Criteria

1. Pipeline tests pass.
2. Existing trainer/data ergonomics tests remain green.

### Phase 42: Collate Registry and Schema Composition

#### Problem

Collate logic is powerful but repetitive across train/validation flows when the same mapping/callable schemas are reused.

#### Deliverables

1. Add trainer-level collate registry:
   - `register_collate(name, spec = nil, &block)`
   - support named reuse in `collate:` / `validation_collate:`
2. Support schema composition helpers for named collates (for example, extending base mapping selectors).
3. Keep built-in schemas (`:x`, `:xy`) backward compatible.

#### Red (tests first)

1. Add failing trainer unit tests for registering and resolving named collate schemas.
2. Add failing tests for train/validation parity using shared named collates.

#### Green (minimum implementation)

1. Add collate registry storage and lookup path in `lib/mlx/dsl/trainer.rb`.
2. Route collate normalization through registry before existing dispatch.

#### Exit Criteria

1. Named collate registry tests pass.
2. Existing collate behavior remains green.

### Phase 43: Ordered and Conditional Hook Middleware

#### Problem

Hooks currently execute in registration order only, with no built-in scheduling controls (`every N`, once-only, or priority ordering).

#### Deliverables

1. Extend hook registration to support options:
   - `priority:`
   - `every:`
   - `once:`
   - optional `if:` predicate
2. Ensure deterministic ordering for hooks with mixed priorities.
3. Preserve existing `on` and shorthand hook APIs without options.

#### Red (tests first)

1. Add failing trainer/train-step unit tests for hook ordering by priority.
2. Add failing tests for `every:` and `once:` scheduling semantics.
3. Add failing tests for conditional hook predicates.

#### Green (minimum implementation)

1. Add hook wrapper normalization and ordered execution in `lib/mlx/dsl/train_step.rb` and `lib/mlx/dsl/trainer.rb`.
2. Keep no-option hooks on current behavior path.

#### Exit Criteria

1. Hook ordering/scheduling tests pass.
2. Existing hook consumers remain green.

### Phase 44: Model Introspection and Debug Ergonomics

#### Problem

As DSL graphs get more dynamic, users need fast visibility into module composition, parameter paths, and trainable counts without custom scripts.

#### Deliverables

1. Add model introspection helpers:
   - `summary`
   - `parameter_count`
   - `trainable_parameter_count`
   - `parameter_paths(matcher: nil)`
2. Ensure output reflects DSL-built graphs (including callable/composed modules).
3. Provide machine-friendly summary payloads (hash) and human-readable formatting.

#### Red (tests first)

1. Add failing DSL tests for parameter/path counts on composed models.
2. Add failing tests for matcher-filtered path reporting.

#### Green (minimum implementation)

1. Add introspection helpers in `lib/mlx/dsl/model_mixin.rb`.
2. Reuse existing tree flatten utilities for consistent path semantics.

#### Exit Criteria

1. Introspection tests pass.
2. Existing freeze/unfreeze and optimizer-group behavior stays green.

### Phase 45: Reproducible Run Bundle and Resume Metadata Standardization

#### Problem

Experiment reproducibility remains fragmented across ad-hoc metadata, making restart/debug workflows less reliable over long-running iterations.

#### Deliverables

1. Add trainer run bundle export (JSON) containing:
   - fit report
   - trainer config (`monitor`, reducer, limits, resume source kind)
   - checkpoint metadata snapshot
2. Add versioned metadata schema key for trainer-resume fields.
3. Add helper for loading run bundle metadata into `resume_from` callable/hash flows.

#### Red (tests first)

1. Add failing integration tests for run bundle export payload shape.
2. Add failing tests for resume compatibility using exported metadata.

#### Green (minimum implementation)

1. Add run bundle serialization helper in `lib/mlx/dsl/trainer.rb`.
2. Document schema version and compatibility expectations.

#### Exit Criteria

1. Run bundle tests pass.
2. Resume and checkpoint compatibility tests remain green.

### Phase 46: Multi-Base Collate Schema Composition

#### Problem

Named collate composition currently supports only a single `extends:` base, which makes shared train/eval schema layering repetitive in real projects.

#### Deliverables

1. Allow `register_collate(..., extends: [...])` with ordered base names.
2. Compose base schemas in-order before applying the overlay schema.
3. Preserve clear unknown-base errors for each missing base name.

#### Red (tests first)

1. Add failing trainer unit test proving multi-base `extends` composition order.
2. Add failing trainer unit test proving unknown base names raise explicit errors in multi-base mode.

#### Green (minimum implementation)

1. Update `Trainer#register_collate` to normalize `extends` as single-name or array.
2. Compose ordered base schemas before overlay merge in `lib/mlx/dsl/trainer.rb`.

#### Exit Criteria

1. Multi-base collate tests pass.
2. Existing named collate behavior remains green.

### Phase 47: Dynamic Per-Epoch Loop Limits

#### Problem

`limit:` and `validation_limit:` are static values today, forcing external wrappers for common Ruby dynamic scheduling patterns.

#### Deliverables

1. Allow `limit:` to accept callables resolved per epoch.
2. Allow `validation_limit:` to accept callables resolved per epoch.
3. Support callable signatures with runtime context (`epoch`, `kind`, `trainer`).
4. Validate negative/invalid callable returns with clear errors.

#### Red (tests first)

1. Add failing trainer unit test for callable train limits varying by epoch.
2. Add failing trainer unit test for callable validation limits varying by epoch.

#### Green (minimum implementation)

1. Add loop-limit resolution helper in `lib/mlx/dsl/trainer.rb`.
2. Resolve per-epoch limit values before train and validation iteration loops.

#### Exit Criteria

1. Callable limit tests pass.
2. Existing integer limit behavior remains unchanged.

### Phase 48: Callable Checkpoint Path Builders

#### Problem

Checkpoint naming currently depends on string templates only, which underuses Ruby’s dynamic DSL style for contextual path generation.

#### Deliverables

1. Allow `checkpoint_path:` to accept a callable path builder.
2. Provide path-builder context (`epoch`, `next_epoch`, monitor fields, `trainer`, `model`, `optimizer`).
3. Validate callable return values as string-compatible paths.
4. Preserve existing `%{...}` template support.

#### Red (tests first)

1. Add failing trainer unit test proving callable checkpoint paths receive full runtime context.

#### Green (minimum implementation)

1. Extend checkpoint-path resolver in `lib/mlx/dsl/trainer.rb` for callable dispatch.
2. Keep template interpolation fallback for string paths.

#### Exit Criteria

1. Callable checkpoint path tests pass.
2. Existing template behavior remains green.

### Phase 49: Run Bundle Resume Source Autodetection

#### Problem

Resume flows currently require manual conversion from run bundles to checkpoint metadata payloads, adding avoidable boilerplate.

#### Deliverables

1. Accept run-bundle hashes directly in `resume_from:`.
2. Accept run-bundle JSON paths directly in `resume_from:`.
3. Route detected run bundles through `resume_payload_from_bundle` automatically.
4. Preserve existing path-based checkpoint resume behavior for non-bundle paths.

#### Red (tests first)

1. Add failing trainer unit test for run-bundle hash resume.
2. Add failing trainer unit test for run-bundle path resume.

#### Green (minimum implementation)

1. Add run-bundle source detection helpers in `lib/mlx/dsl/trainer.rb`.
2. Normalize resume source before checkpoint-loading fallback.

#### Exit Criteria

1. Run-bundle resume tests pass.
2. Existing inline/hash/callable/path resume behavior remains green.

### Phase 50: Index-Aware Dataset Pipeline Transforms

#### Problem

`Data::Pipeline#map` and `#filter` currently receive only the batch item, limiting Ruby-idiomatic index-aware transform logic.

#### Deliverables

1. Add index-aware callable invocation for pipeline `map`.
2. Add index-aware callable invocation for pipeline `filter`.
3. Support positional and keyword signatures with `item`, `index`, and `pipeline` context.

#### Red (tests first)

1. Add failing pipeline unit test for positional `(item, index)` mapping.
2. Add failing pipeline unit test for keyword `index:` filtering.

#### Green (minimum implementation)

1. Add signature-aware callable dispatcher for pipeline transforms in `lib/mlx/dsl/data_pipeline.rb`.
2. Route `map`/`filter` through the new dispatcher while preserving lazy iteration semantics.

#### Exit Criteria

1. Index-aware pipeline tests pass.
2. Existing pipeline chaining/laziness tests remain green.

### Phase 51: Trainer Fit Presets and Defaults Registry

#### Problem

`fit` / `fit_report` calls repeat large keyword argument sets (`monitor`, `reduce`, checkpoint policy, limits, resume policy), creating copy/paste boilerplate across scripts.

#### Deliverables

1. Add trainer-level preset registry:
   - `register_fit_preset(name, **defaults)`
   - `fit_with(name, dataset, **overrides)`
   - `fit_report_with(name, dataset, **overrides)`
2. Add immutable trainer defaults helper:
   - `with_fit_defaults(**defaults)` returning a configured trainer wrapper/clone.
3. Merge precedence:
   - explicit call overrides > preset defaults > trainer defaults > current method defaults.
4. Keep existing `fit` / `fit_report` APIs unchanged.

#### Red (tests first)

1. Add failing trainer unit tests for preset registration and `fit_with` / `fit_report_with` execution.
2. Add failing tests for precedence/merge semantics across trainer defaults, preset defaults, and call overrides.
3. Add failing tests proving existing direct `fit` usage remains unaffected.

#### Green (minimum implementation)

1. Add preset/default storage and merge normalization in `lib/mlx/dsl/trainer.rb`.
2. Route `fit_with` / `fit_report_with` through existing `fit` execution path.

#### Exit Criteria

1. Preset/default tests pass.
2. Existing trainer behavior remains green.

### Phase 52: Declarative Batch Schema and Auto-Collate

#### Problem

Users repeatedly write equivalent `collate`/`validation_collate` mappings for common `(x, y)` and nested hash batches.

#### Deliverables

1. Add declarative batch schema API:
   - `batch_schema(spec)` at trainer level
   - optional split-specific schemas (`train_schema`, `validation_schema`).
2. Add auto-collate mode:
   - `collate: :auto` / `validation_collate: :auto` using declared schema or inferred defaults.
3. Inference behavior for common batch shapes:
   - hash with `x`/`y` keys (symbol or string)
   - two-item arrays -> `{x:, y:}`
4. Keep explicit collate specs taking precedence over schema/auto behavior.

#### Red (tests first)

1. Add failing trainer unit tests for schema-driven auto-collation.
2. Add failing tests for split-specific schema overrides.
3. Add failing tests for precedence when explicit `collate` is provided.

#### Green (minimum implementation)

1. Add schema storage and auto-collate resolver in `lib/mlx/dsl/trainer.rb`.
2. Reuse existing collate mapping and callable dispatch internals where possible.

#### Exit Criteria

1. Auto-collate/schema tests pass.
2. Existing manual collate behavior remains green.

### Phase 53: Reusable Dataflow Specs (Collate + Transform + Limits)

#### Problem

Even with collate reuse, users still repeat the same train/validation loop wiring (`collate`, transforms, limits, reducers) across runs.

#### Deliverables

1. Add composable dataflow profiles:
   - `register_dataflow(name, train: {...}, validation: {...})`
   - `use_dataflow(name, **overrides)` on fit calls.
2. Dataflow fields should support existing dynamic callables:
   - `collate`, `transform`, `limit`, `reduce` / `validation_reduce`.
3. Support profile inheritance/composition:
   - `extends:` for dataflow profiles (same merge semantics as fit presets).
4. Keep direct keyword usage fully backward compatible.

#### Red (tests first)

1. Add failing trainer unit tests for applying named dataflow profiles.
2. Add failing tests for profile inheritance and override precedence.
3. Add failing tests proving direct per-call kwargs override profile values.

#### Green (minimum implementation)

1. Add dataflow registry + profile merge in `lib/mlx/dsl/trainer.rb`.
2. Resolve profile-derived fit kwargs before main fit execution.

#### Exit Criteria

1. Dataflow profile tests pass.
2. Existing dataset/collate/transform behavior remains green.

### Phase 54: Stack/Repeat Builder Macros

#### Problem

Model declaration blocks repeat similar layer sequences manually (e.g., `linear + relu + dropout` N times), adding noise and error-prone copy/paste.

#### Deliverables

1. Add builder repetition helpers:
   - `repeat_layers(count) { |i| ... }`
   - `stack(count, layer_class = nil, *args, **kwargs, &block)` for common repeated patterns.
2. Ensure repeated entries are normalized through existing module/callable resolution paths.
3. Support index-aware blocks for dynamic dimensions (`i`-dependent construction).
4. Preserve existing `sequential`/`branch` semantics and output module tracking.

#### Red (tests first)

1. Add failing graph unit tests for repeated layer construction and index-aware block behavior.
2. Add failing tests proving module tracking and parameter paths remain correct for repeated stacks.

#### Green (minimum implementation)

1. Implement repeat/stack helpers in `lib/mlx/dsl/builder.rb`.
2. Reuse existing composition normalization internals for consistency.

#### Exit Criteria

1. Stack/repeat tests pass.
2. Existing graph/builder tests remain green.

### Phase 55: Hook and Metric Packs

#### Problem

Hook instrumentation and monitor metric setup are frequently duplicated between experiments (`logging`, `checkpoint telemetry`, `early-stop traces`).

#### Deliverables

1. Add reusable hook packs:
   - `register_hook_pack(name) { ... }`
   - `use_hook_pack(name, **options)`
2. Add reusable metric packs:
   - `register_metric(name, callable = nil, &block)`
   - reference metric by name in `fit_report(monitor:, metric:)`.
3. Allow hook/metric packs to receive runtime context and user options.
4. Preserve direct inline `on` hooks and callable `metric:` behavior.

#### Red (tests first)

1. Add failing trainer unit tests for applying named hook packs.
2. Add failing tests for named metric registration and monitor integration.
3. Add failing tests for per-use options/context propagation.

#### Green (minimum implementation)

1. Add hook/metric registries in `lib/mlx/dsl/trainer.rb`.
2. Resolve named packs/metrics through existing emit and monitor execution paths.

#### Exit Criteria

1. Hook/metric pack tests pass.
2. Existing hook and metric behavior remains green.

### Phase 56: Task-Level Training API (`fit_task`)

#### Problem

Training setup still repeats task boilerplate (`loss`, `monitor`, default collate shape, metric wiring) for common workflows like classification and regression.

#### Deliverables

1. Add task-level fit entrypoints:
   - `fit_task(task, dataset, **kwargs)`
   - `fit_task_report(task, dataset, **kwargs)`
2. Add built-in task presets:
   - `:classification`
   - `:regression`
   - `:language_modeling`
3. Task presets should provide sane defaults for:
   - loss callable
   - `monitor` / `monitor_mode`
   - common collate schema assumptions
4. Preserve current `fit` / `fit_report` and custom loss-block workflows unchanged.

#### Red (tests first)

1. Add failing trainer unit tests for built-in classification task wiring.
2. Add failing tests for task defaults being overrideable per call.
3. Add failing tests proving non-task fit flows remain unchanged.

#### Green (minimum implementation)

1. Add task registry/resolution in `lib/mlx/dsl/trainer.rb`.
2. Route `fit_task` APIs through existing fit execution path via normalized kwargs/loss callable.

#### Exit Criteria

1. Task-level fit tests pass.
2. Existing trainer APIs remain green.

### Phase 57: Signature and KeyPath Auto-Binding for Batches

#### Problem

Users still write repetitive `collate` mappings to bridge dataset batch shapes into loss/train-step signatures (`x:`, `y:`, nested targets).

#### Deliverables

1. Add auto-binding option for train and validation:
   - `bind:`
   - `validation_bind:`
2. Support binding modes:
   - argument-name inference from loss/train-step signature
   - explicit key-path mappings (for example, `{ x: [:input, :x], y: [:target, 0] }`)
3. Keep duplicate-key and missing-key diagnostics explicit and context-rich.
4. Preserve explicit `collate` precedence over auto-binding.

#### Red (tests first)

1. Add failing trainer unit tests for signature-inferred binding on hash batches.
2. Add failing tests for nested key-path binding.
3. Add failing tests for precedence and error diagnostics.

#### Green (minimum implementation)

1. Add bind normalization and extraction helpers in `lib/mlx/dsl/trainer.rb`.
2. Integrate binding into batch preparation before train/validation dispatch.

#### Exit Criteria

1. Auto-binding tests pass.
2. Existing collate and batch dispatch flows remain green.

### Phase 58: Unified Experiment DSL (`experiment do ... end`)

#### Problem

Experiment scripts still spread setup across model/trainer/optimizer/dataflow/checkpoint blocks, requiring repeated orchestration scaffolding.

#### Deliverables

1. Add top-level experiment DSL helper:
   - `MLX::DSL.experiment(name = nil) { ... }`
2. Support declarative sections:
   - `model`
   - `optimizer`
   - `trainer`
   - `dataflow` / datasets
   - `artifacts` / resume settings
3. Return a runnable experiment object with:
   - `run`
   - `report`
   - `save_run_bundle`
4. Keep all existing lower-level APIs available and unchanged.

#### Red (tests first)

1. Add failing integration test for end-to-end experiment declaration and execution.
2. Add failing tests for section override precedence and explicit object injection.

#### Green (minimum implementation)

1. Add experiment builder/runtime under `lib/mlx/dsl/`.
2. Reuse existing trainer/model helpers instead of duplicating training logic.

#### Exit Criteria

1. Experiment DSL tests pass.
2. Existing DSL entrypoints remain green.

### Phase 59: Dataset Split Plan DSL

#### Problem

Train/validation/test split wiring still requires repeated ad-hoc lambdas and transform plumbing across scripts.

#### Deliverables

1. Add split-plan DSL:
   - `splits do ... end`
   - `train`, `validation`, `test` declarations
2. Support shared and split-specific transforms/collate/limits.
3. Provide reusable split plan objects consumable by trainer fit/report APIs.
4. Preserve compatibility with raw enumerable and factory datasets.

#### Red (tests first)

1. Add failing trainer unit tests for split plan train/validation consumption.
2. Add failing tests for shared transform inheritance plus split overrides.
3. Add failing tests for backward compatibility with current dataset inputs.

#### Green (minimum implementation)

1. Add split plan object and resolver in `lib/mlx/dsl/`.
2. Integrate plan expansion into trainer call paths as optional sugar.

#### Exit Criteria

1. Split-plan tests pass.
2. Existing dataset and transform behaviors remain green.

### Phase 60: Artifact Policy DSL (Checkpoints and Run Bundles)

#### Problem

Checkpoint/run-bundle lifecycle policies (`latest`, `best`, retention count, naming, resume target) are still configured manually per script.

#### Deliverables

1. Add declarative artifact policy API:
   - checkpoint strategy (`save_latest`, `save_best`, `save_every`)
   - retention (`keep_last_n`)
   - resume strategy (`:latest`, `:best`, `:path`, callable)
2. Add run-bundle export policy toggles and output conventions.
3. Keep policy evaluation deterministic and report-visible.
4. Preserve current direct checkpoint/run-bundle APIs.

#### Red (tests first)

1. Add failing trainer unit/integration tests for retention and strategy behavior.
2. Add failing tests for resume strategy resolution from policy state.
3. Add failing tests for policy metadata appearing in reports/bundles.

#### Green (minimum implementation)

1. Add artifact policy object and enforcement hooks in `lib/mlx/dsl/trainer.rb`.
2. Route checkpoint/bundle paths through policy resolver while keeping existing explicit options.

#### Exit Criteria

1. Artifact policy tests pass.
2. Existing checkpoint and run-bundle behavior remains green.

## Immediate Implementation Scope

Implemented in this execution stream: Phases 19-60.
Next planning queue: Phase 61+.

Implementation approach for each phase remains strict red/green sequencing:

1. Red tests for each phase.
2. Minimum green code changes.
3. Targeted suite run, then broader DSL regression run.
