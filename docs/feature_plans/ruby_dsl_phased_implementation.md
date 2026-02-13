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
3. Add initial tests in `test/dsl_test.rb`.

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

## Red/Green Execution Plan

1. Add tests for each DSL behavior in `test/dsl_test.rb` (red).
2. Implement minimum code in `lib/mlx/dsl/**` to pass tests (green).
3. Refactor and tighten internals while maintaining passing tests.
4. Run targeted suite first, then broader suite.
