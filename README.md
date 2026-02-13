# MLX Ruby

[![Build and Test](https://github.com/skryl/mlx-ruby/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/skryl/mlx-ruby/actions/workflows/build_and_test.yml)
[![RubyGems](https://img.shields.io/gem/v/mlx.svg)](https://rubygems.org/gems/mlx)
[![Documentation](https://img.shields.io/badge/docs-MLX%20Ruby-blue)](https://skryl.github.io/mlx-ruby)

[Full Documentation](https://skryl.github.io/mlx-ruby)

Ruby bindings for [MLX](https://github.com/ml-explore/mlx): a NumPy-like array framework for machine learning.

This repository packages:

- A native Ruby extension backed by the upstream C++ MLX runtime.
- Ruby APIs that mirror the core MLX package layout: `MLX::Core`, `MLX::NN`, `MLX::Optimizers`, `MLX::Utils`, and distributed helpers.
- Parity and contract tooling used to keep the Ruby surface aligned with upstream MLX behavior.

## Highlights

- Lazy arrays and dynamic graph construction.
- Function transforms (`grad`, `value_and_grad`, `vmap`, `jvp`, `vjp`, `compile`, and more).
- Neural-network layers, losses, initialization, and optimizers.
- Device-aware execution (CPU/GPU support exposed through MLX), including full Metal support on Apple silicon.
- Extensive parity testing and generated report artifacts.

## Requirements

- Ruby `>= 3.1` (from `mlx.gemspec`).
- Git (with submodule support).
- CMake `>= 3.25`.
- A C++20-capable toolchain.
- macOS: Xcode command-line tools and CMake.
- Linux: standard build tools plus BLAS/LAPACK headers (CI uses `build-essential cmake libopenblas-dev liblapacke-dev`).

## Installation

### Install from RubyGems

```bash
gem install mlx
```

### Install from source (recommended for development)

```bash
git clone --recurse-submodules https://github.com/skryl/mlx-ruby.git
cd mlx-ruby
bundle install
bundle exec rake test
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

Build and install a local gem:

```bash
gem build mlx.gemspec
gem install ./mlx-*.gem
```

Use from another project via local path:

```ruby
gem "mlx", path: "/absolute/path/to/mlx-ruby"
```

### Verify installation

```bash
bundle exec ruby -e 'require "mlx"; puts MLX::VERSION; puts "native=#{MLX.native_available?}"'
```

## Examples

For end-to-end examples, see [`skryl/mlx-ruby-examples`](https://github.com/skryl/mlx-ruby-examples).

- [Transformer](https://github.com/skryl/mlx-ruby-examples/tree/main/transformer_lm)
- [LLaMA](https://github.com/skryl/mlx-ruby-examples/tree/main/llms/llama)
- [LoRA](https://github.com/skryl/mlx-ruby-examples/tree/main/lora)
- [Stable Diffusion](https://github.com/skryl/mlx-ruby-examples/tree/main/stable_diffusion)
- [Whisper](https://github.com/skryl/mlx-ruby-examples/tree/main/whisper)

## Quickstart

### Arrays and lazy execution

```ruby
require "mlx"

mx = MLX::Core
x = mx.array([1.0, 2.0, 3.0], mx.float32)
y = mx.sqrt(x + 1.0)

mx.eval(y)         # force materialization
p y.to_a           # => [1.414..., 1.732..., 2.0]
```

### Minimal trainable module

```ruby
require "mlx"

mx = MLX::Core

class LinearRegressor < MLX::NN::Module
  def initialize
    super()
    self.linear = MLX::NN::Linear.new(3, 1)
  end

  def call(x)
    linear.call(x)
  end
end

model = LinearRegressor.new
optimizer = MLX::Optimizers::AdamW.new(learning_rate: 1e-2)

loss_and_grad = MLX::NN.value_and_grad(
  model,
  lambda do |inputs, targets|
    diff = model.call(inputs) - targets
    mx.mean(diff * diff)
  end
)

x = mx.array([[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]], mx.float32)
y = mx.array([[1.0], [0.0]], mx.float32)

5.times do |step|
  loss, grads = loss_and_grad.call(x, y)
  optimizer.update(model, grads)
  mx.eval(loss, model.parameters, optimizer.state)
  puts "step=#{step} loss=#{loss.item}"
end
```

Important: when defining `MLX::NN::Module`, register trainable arrays/submodules with `self.<name> = ...` (not only `@ivar = ...`) so they are tracked in `parameters` and optimized correctly.

### Small CNN (single training step)

```ruby
require "mlx"

mx = MLX::Core

class SmallCNN < MLX::NN::Module
  def initialize(num_classes: 10)
    super()
    self.conv1 = MLX::NN::Conv2d.new(1, 16, 3, padding: 1)
    self.conv2 = MLX::NN::Conv2d.new(16, 32, 3, padding: 1)
    self.relu = MLX::NN::ReLU.new
    self.pool = MLX::NN::MaxPool2d.new(2, stride: 2)
    self.fc1 = MLX::NN::Linear.new(32 * 7 * 7, 64)
    self.fc2 = MLX::NN::Linear.new(64, num_classes)
  end

  def call(x)
    y = pool.call(relu.call(conv1.call(x)))
    y = pool.call(relu.call(conv2.call(y)))
    y = MLX::Core.reshape(y, [y.shape[0], 32 * 7 * 7])
    y = relu.call(fc1.call(y))
    fc2.call(y)
  end
end

model = SmallCNN.new(num_classes: 10)
optimizer = MLX::Optimizers::Adam.new(learning_rate: 1e-3)

loss_and_grad = MLX::NN.value_and_grad(
  model,
  lambda do |images, labels|
    logits = model.call(images)
    MLX::NN.cross_entropy(logits, labels, reduction: "mean")
  end
)

images = mx.random_uniform([4, 28, 28, 1], 0.0, 1.0, mx.float32)
labels = mx.array([1, 3, 4, 7], mx.int32)

loss, grads = loss_and_grad.call(images, labels)
optimizer.update(model, grads)
mx.eval(loss, model.parameters, optimizer.state)
puts "cnn_loss=#{loss.item}"
```

### Karpathy-style nano GPT (single training step)

```ruby
require "mlx"

mx = MLX::Core
vocab_size = 65
seq_len = 32
batch_size = 4
dims = 128
heads = 4
layers = 2

class NanoGpt < MLX::NN::Module
  def initialize(vocab_size:, seq_len:, dims:, heads:, layers:)
    super()
    self.token_embedding = MLX::NN::Embedding.new(vocab_size, dims)
    self.pos_embedding = MLX::NN::Embedding.new(seq_len, dims)
    self.blocks = Array.new(layers) do
      MLX::NN::TransformerEncoderLayer.new(
        dims,
        heads,
        mlp_dims: dims * 4,
        dropout: 0.0,
        norm_first: true
      )
    end
    self.norm = MLX::NN::LayerNorm.new(dims)
    self.head = MLX::NN::Linear.new(dims, vocab_size)
    @causal_mask = MLX::NN::MultiHeadAttention.create_additive_causal_mask(seq_len)
  end

  def call(input_ids)
    positions = MLX::Core.arange(0, input_ids.shape[1], 1, MLX::Core.int32)
    hidden = MLX::Core.add(token_embedding.call(input_ids), pos_embedding.call(positions))
    blocks.each { |block| hidden = block.call(hidden, @causal_mask) }
    head.call(norm.call(hidden))
  end
end

tokens = Array.new(batch_size) { Array.new(seq_len) { rand(vocab_size) } }
targets = tokens.map { |row| row[1..] + [0] }

input_ids = mx.array(tokens, mx.int32)
target_ids = mx.array(targets, mx.int32)

model = NanoGpt.new(vocab_size: vocab_size, seq_len: seq_len, dims: dims, heads: heads, layers: layers)
optimizer = MLX::Optimizers::AdamW.new(learning_rate: 1e-3)

loss_and_grad = MLX::NN.value_and_grad(
  model,
  lambda do |ids, labels|
    logits = model.call(ids)
    logits2d = MLX::Core.reshape(logits, [batch_size * seq_len, vocab_size])
    labels1d = MLX::Core.reshape(labels, [batch_size * seq_len])
    MLX::NN.cross_entropy(logits2d, labels1d, reduction: "mean")
  end
)

loss, grads = loss_and_grad.call(input_ids, target_ids)
optimizer.update(model, grads)
mx.eval(loss, model.parameters, optimizer.state)
puts "nanogpt_loss=#{loss.item}"
```

### Ruby DSL (experimental ergonomic layer)

`MLX::DSL` provides Ruby-style declarations on top of the existing `MLX::NN::Module` behavior.

```ruby
require "mlx"

class Mlp < MLX::DSL::Model
  option :in_dim
  option :hidden_dim, default: 128
  option :out_dim

  layer :net do
    sequential do
      linear in_dim, hidden_dim
      relu
      linear hidden_dim, out_dim
    end
  end

  def call(x)
    net.call(x)
  end
end

model = Mlp.new(in_dim: 32, out_dim: 10)
```

You can also mix the DSL into existing `MLX::NN::Module` subclasses:

```ruby
class Block < MLX::NN::Module
  include MLX::DSL::ModelMixin

  option :dims, default: 64
  layer(:proj) { linear dims, dims, bias: false }
  layer(:norm) { layer_norm dims }

  def call(x)
    norm.call(proj.call(x))
  end
end
```

`train_step` wraps `value_and_grad` and optimizer updates:

```ruby
optimizer = MLX::Optimizers::AdamW.new(learning_rate: 1e-3)

step = model.train_step(optimizer: optimizer, clip_grad_norm: 1.0) do |x:, y:|
  logits = model.call(x)
  MLX::NN.cross_entropy(logits, y, reduction: "mean")
end

step.after_step { |ctx| puts "step=#{ctx[:step]} loss=#{ctx[:loss].item}" }
loss = step.call(x: batch_x, y: batch_y)
```

A small trainer wrapper is also available:

```ruby
trainer = model.trainer(optimizer: optimizer) do |x:, y:|
  logits = model.call(x)
  MLX::NN.cross_entropy(logits, y, reduction: "mean")
end

trainer.before_epoch { |ctx| puts "epoch=#{ctx[:epoch]}" }
trainer.after_batch { |ctx| puts "batch=#{ctx[:batch_index]} loss=#{ctx[:loss_value]}" }

train_data = ->(epoch, kind:) { shuffled_batches_for(epoch, split: kind) }
validation_data = ->(epoch:) { heldout_batches_for(epoch) }

losses = trainer.fit(train_data, epochs: 2)

report = trainer.fit_report(
  train_data,
  epochs: 5,
  reduce: :mean,
  strict_data_reuse: true,
  train_transform: ->(batch, epoch:, batch_index:) { collate_train(batch, epoch: epoch, batch_index: batch_index) },
  validation_data: validation_data,
  validation_reduce: :mean,
  validation_transform: ->(batch, epoch:) { collate_eval(batch, epoch: epoch) },
  monitor: :val_loss,
  monitor_mode: :min,
  checkpoint_path: "checkpoints/epoch-%{epoch}-monitor-%{monitor}.bin",
  save_best: true,
  keep_losses: false,
  metadata: { "run" => "exp-42" }
)

puts "#{report['monitor_name']}=#{report['best_metric']}"
```

Runnable DSL examples:

- `examples/dsl/streaming_factory.rb`
- `examples/dsl/validation_monitor.rb`
- `examples/dsl/memory_friendly_reporting.rb`

For non-linear graph composition, the DSL also supports branch/merge helpers:

```ruby
class ResidualHead < MLX::DSL::Model
  option :dims, default: 64

  layer :merge do
    concat(axis: -1) do
      identity
      residual do
        linear dims, dims
      end
    end
  end

  def call(x)
    merge.call(x)
  end
end
```

Parameter-group optimizers:

```ruby
grouped = model.optimizer_groups do
  group(/^encoder\./) { MLX::Optimizers::AdamW.new(learning_rate: 1e-4) }
  group(nil) { MLX::Optimizers::SGD.new(learning_rate: 1e-2) }
end
```

## Device selection

Default device is chosen at load time. You can override it:

```bash
MLX_DEFAULT_DEVICE=gpu bundle exec ruby your_script.rb
```

Supported values are `cpu`, `gpu`, or `metal` (`metal` maps to GPU selection when available). `DEVICE` is also accepted as a fallback environment variable.

## Development

### Build native extension

```bash
bundle exec rake build
```

### Clean native build artifacts

```bash
bundle exec rake clean
```

### Run tests

```bash
bundle exec rake test
```

Strict mode (per-file timeout):

```bash
MLX_STRICT_TESTS=1 MLX_TEST_TIMEOUT=30 bundle exec rake test
```

### Benchmarks (Ruby vs Python implementations)

List tasks:

```bash
bundle exec rake -T
```

Run one benchmark:

```bash
bundle exec rake benchmark:transformer
```

Run all benchmark suites:

```bash
bundle exec rake benchmark:all
```

Benchmark environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DEVICE` | `gpu` | Compute device (`cpu`, `gpu`, or `metal`) |
| `ITERATIONS` | `50` | Timed iterations |
| `WARMUP` | `10` | Warmup iterations |
| `BATCH` | `8` | Batch size |
| `SEQUENCE_LENGTH` | `128` | Source sequence length |
| `TARGET_SEQUENCE_LENGTH` | `64` | Target sequence length |
| `DIMENSIONS` | `256` | Model width |
| `HEADS` | `8` | Attention heads |
| `LAYERS` | `4` | Number of layers |
| `PYTHON` | `python3` | Python executable for cross-language comparison |

### Performance

MLX Ruby has full Metal support through the upstream MLX runtime. On Apple silicon, use `DEVICE=metal` (or `DEVICE=gpu`) to run on Metal.

The table below is from:

```bash
bundle exec rake benchmark:all DEVICE=gpu ITERATIONS=1000 WARMUP=50 BATCH=2 SEQUENCE_LENGTH=32 TARGET_SEQUENCE_LENGTH=16 DIMENSIONS=64 HEADS=4 LAYERS=2
bundle exec rake benchmark:all DEVICE=cpu ITERATIONS=1000 WARMUP=50 BATCH=2 SEQUENCE_LENGTH=32 TARGET_SEQUENCE_LENGTH=16 DIMENSIONS=64 HEADS=4 LAYERS=2
```

Updated with reversed ratios (`Python/Ruby`), so `< 1x` means Ruby is slower.

| Model | Ruby CPU (ms) | Python CPU (ms) | Python/Ruby CPU (x) | Ruby GPU (ms) | Python GPU (ms) | Python/Ruby GPU (x) |
| --- | --- | --- | --- | --- | --- | --- |
| transformer | 1.771 | 1.242 | 0.70x | 1.315 | 0.830 | 0.63x |
| cnn | 4.070 | 4.271 | 1.05x | 1.113 | 0.958 | 0.86x |
| mlp | 0.168 | 0.258 | 1.54x | 0.313 | 0.218 | 0.70x |
| rnn | 0.919 | 0.532 | 0.58x | 1.168 | 0.644 | 0.55x |

### Build docs

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r docs/requirements.txt
cd docs
doxygen
make html
```

Built docs are placed under `docs/build/html`.

## Repository layout

- `lib/`: Ruby API surface and compatibility layer.
- `ext/mlx/`: native extension (`extconf.rb`, C++ bridge).
- `mlx/`: upstream MLX submodule.
- `test/`: unit/parity tests (including large parity phase suite).
- `tools/parity/`: report and contract generators.
- `tasks/benchmark_task.rb`: benchmark harness.
- `docs/`: Sphinx + Doxygen documentation sources.

## Troubleshooting

- `missing MLX include dir`: initialize submodules (`git submodule update --init --recursive`).
- Native extension does not load: rebuild manually:

```bash
cd ext/mlx
ruby extconf.rb
make -j4
```

- On Apple silicon, verify native architecture:

```bash
ruby -e 'require "rbconfig"; puts RbConfig::CONFIG["host_cpu"]'
```

- If CMake configure fails intermittently, rerun `ruby extconf.rb`; the build script already includes a clean-retry path.

## Contributing

- Open pull requests against this repository.
- Keep parity artifacts in `tools/parity/reports/` in sync with tool/script changes.
- Follow upstream MLX contributor guidance where applicable: [mlx/CONTRIBUTING.md](https://github.com/ml-explore/mlx/blob/main/CONTRIBUTING.md).

CI currently runs on `ubuntu-22.04` and `macos-14` with Ruby `3.3`.

## License

`mlx` gem is distributed under the MIT license (see `LICENSE`).
