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
- Device-aware execution (CPU/GPU support exposed through MLX).
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

## Device selection

Default device is chosen at load time. You can override it:

```bash
MLX_DEFAULT_DEVICE=gpu bundle exec ruby your_script.rb
```

Supported values are `cpu`, `gpu`, or `metal` (`metal` maps to GPU selection when available). `DEVICE` is also accepted as a fallback environment variable.

## Development

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

## Citing MLX

If MLX is useful in your research, you can cite:

```text
@software{mlx2023,
  author = {Awni Hannun and Jagrit Digani and Angelos Katharopoulos and Ronan Collobert},
  title = {{MLX}: Efficient and flexible machine learning on Apple silicon},
  url = {https://github.com/ml-explore},
  version = {0.0},
  year = {2023},
}
```

## License

`mlx` gem is distributed under the MIT license (see `mlx/LICENSE`).
