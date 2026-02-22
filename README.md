# MLX Ruby

[![Build and Test](https://github.com/skryl/mlx-ruby/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/skryl/mlx-ruby/actions/workflows/build_and_test.yml)
[![RubyGems](https://img.shields.io/gem/v/mlx.svg)](https://rubygems.org/gems/mlx)
[![Documentation](https://img.shields.io/badge/docs-MLX%20Ruby-blue)](https://skryl.github.io/mlx-ruby)

## Index

- [Full Docs](https://skryl.github.io/mlx-ruby)
- [Examples](https://github.com/skryl/mlx-ruby-examples)
- [Ruby DSL Docs](https://skryl.github.io/mlx-ruby/ruby_dsl/index.html)
- [ONNX/WebGPU Docs](https://skryl.github.io/mlx-ruby/ruby/export.html#onnx-webgpu-support)
- [WebGPU Demo](https://skryl.github.io/mlx-ruby/demo/)

## About

Ruby bindings for [MLX](https://github.com/ml-explore/mlx): a NumPy-like array framework for machine learning.

This repository packages:

- A native Ruby extension backed by the upstream C++ MLX runtime.
- Ruby APIs for `MLX::Core`, `MLX::NN`, `MLX::Optimizers`, `MLX::Utils`,
  distributed helpers, and `MLX::DSL`.
- Graph IR -> ONNX export plus browser harness tooling for WebGPU/wasm paths.
- Parity/contract tooling and benchmark adapters for local models and
  `mlx-ruby-examples`.

## Highlights

- Lazy arrays and dynamic graph construction.
- Function transforms (`grad`, `value_and_grad`, `vmap`, `jvp`, `vjp`, `compile`, and more).
- Neural-network layers, losses, initialization, and optimizers.
- Ruby DSL model/training primitives (`train_step`, trainer, checkpoints,
  data pipelines, experiments).
- Device-aware execution (`cpu`/`gpu`, including Metal-backed GPU on Apple
  silicon when available).
- Graph IR validation, ONNX export, and WebGPU browser harness generation.
- Extensive parity testing (op-level, model fixture, browser harness, and
  examples-submodule coverage).

## Requirements

- Core build/runtime:
  - Ruby `>= 3.3` (from `mlx.gemspec`)
  - Git (with submodule support)
  - CMake `>= 3.25`
  - C++20-capable toolchain
  - macOS: Xcode command-line tools + Metal toolchain
  - Linux: standard build tools + BLAS/LAPACK headers (`build-essential cmake
    libopenblas-dev liblapacke-dev`)
- ONNX export / benchmark helpers:
  - Python 3 with packages from `requirements.txt`
  - `onnx` package available to the interpreter used by `MLX::GraphIR.onnx_json_to_onnx`
- Web smoke/harness workflows:
  - Node.js + `npm` (for `playwright` and `onnxruntime-web`)
- Docs build:
  - Python 3 + `pip`
  - `doxygen`
  - `make`
  - Python deps from `requirements.txt`

## Installation

### macOS prerequisite: install MetalToolchain

On macOS, install the Apple Metal toolchain before installing the gem:

```bash
xcode-select --install
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
xcodebuild -downloadComponent MetalToolchain
```

Optional check:

```bash
xcrun --find metal
```

### Install from RubyGems

```bash
gem install mlx
```

### Install from source (recommended for development)

```bash
git clone --recurse-submodules https://github.com/skryl/mlx-ruby.git
cd mlx-ruby
bundle install
bundle exec rake build
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

Primary end-to-end examples live in
[`skryl/mlx-ruby-examples`](https://github.com/skryl/mlx-ruby-examples).

- [Transformer](https://github.com/skryl/mlx-ruby-examples/tree/main/transformer_lm)
- [LLaMA](https://github.com/skryl/mlx-ruby-examples/tree/main/llms/llama)
- [LoRA](https://github.com/skryl/mlx-ruby-examples/tree/main/lora)
- [Stable Diffusion](https://github.com/skryl/mlx-ruby-examples/tree/main/stable_diffusion)
- [Whisper](https://github.com/skryl/mlx-ruby-examples/tree/main/whisper)

Web demo model/export scripts in this repo are under:

- `examples/web/`


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

class LinearRegressorDsl < MLX::DSL::Model
  option :in_dim, default: 3
  option :out_dim, default: 1
  layer :linear, MLX::NN::Linear, -> { in_dim }, -> { out_dim }

  def call(x)
    linear.call(x)
  end
end

model = LinearRegressorDsl.new
optimizer = MLX::Optimizers::AdamW.new(learning_rate: 1e-2)

step = model.train_step(optimizer: optimizer, sync: :step) do |x:, y:|
  diff = model.call(x) - y
  mx.mean(diff * diff)
end

x = mx.array([[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]], mx.float32)
y = mx.array([[1.0], [0.0]], mx.float32)

5.times do |iter|
  loss = step.call(x: x, y: y)
  puts "step=#{iter} loss=#{loss.item}"
end
```

### Small CNN (single training step)

```ruby
require "mlx"

mx = MLX::Core

class SmallCnnDsl < MLX::DSL::Model
  option :num_classes, default: 10

  layer :features do
    sequential do
      conv2d 1, 16, 3, padding: 1
      relu
      max_pool2d 2, stride: 2
      conv2d 16, 32, 3, padding: 1
      relu
      max_pool2d 2, stride: 2
    end
  end

  layer :classifier do
    sequential do
      fn { |x| MLX::Core.reshape(x, [x.shape[0], 32 * 7 * 7]) }
      linear 32 * 7 * 7, 64
      relu
      linear 64, num_classes
    end
  end

  def call(x)
    classifier.call(features.call(x))
  end
end

model = SmallCnnDsl.new(num_classes: 10)
optimizer = MLX::Optimizers::Adam.new(learning_rate: 1e-3)

step = model.train_step(optimizer: optimizer, sync: :step) do |images:, labels:|
  logits = model.call(images)
  MLX::NN.cross_entropy(logits, labels, reduction: "mean")
end

images = mx.random_uniform([4, 28, 28, 1], 0.0, 1.0, mx.float32)
labels = mx.array([1, 3, 4, 7], mx.int32)

loss = step.call(images: images, labels: labels)
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

class NanoGptDsl < MLX::DSL::Model
  option :vocab_size
  option :seq_len
  option :dims
  option :heads
  option :layers

  layer :token_embedding, MLX::NN::Embedding, -> { vocab_size }, -> { dims }
  layer :pos_embedding, MLX::NN::Embedding, -> { seq_len }, -> { dims }
  layer :encoder, MLX::NN::TransformerEncoder, -> { layers }, -> { dims }, -> { heads },
    mlp_dims: -> { dims * 4 },
    dropout: 0.0,
    norm_first: true
  layer :head, MLX::NN::Linear, -> { dims }, -> { vocab_size }

  def call(input_ids)
    positions = MLX::Core.arange(0, input_ids.shape[1], 1, MLX::Core.int32)
    hidden = MLX::Core.add(token_embedding.call(input_ids), pos_embedding.call(positions))
    mask = MLX::NN::MultiHeadAttention.create_additive_causal_mask(input_ids.shape[1])
    head.call(encoder.call(hidden, mask))
  end
end

tokens = Array.new(batch_size) { Array.new(seq_len) { rand(vocab_size) } }
targets = tokens.map { |row| row[1..] + [0] }

input_ids = mx.array(tokens, mx.int32)
target_ids = mx.array(targets, mx.int32)

model = NanoGptDsl.new(vocab_size: vocab_size, seq_len: seq_len, dims: dims, heads: heads, layers: layers)
optimizer = MLX::Optimizers::AdamW.new(learning_rate: 1e-3)

step = model.train_step(optimizer: optimizer, sync: :step) do |input_ids:, target_ids:|
  logits = model.call(input_ids)
  logits2d = MLX::Core.reshape(logits, [batch_size * seq_len, vocab_size])
  labels1d = MLX::Core.reshape(target_ids, [batch_size * seq_len])
  MLX::NN.cross_entropy(logits2d, labels1d, reduction: "mean")
end

loss = step.call(input_ids: input_ids, target_ids: target_ids)
puts "nanogpt_loss=#{loss.item}"
```

## Device selection

Default device selection runs during `require "mlx"`:

- `MLX_DEFAULT_DEVICE=cpu|gpu|metal`
- fallback: `DEVICE=cpu|gpu|metal`

On systems without Metal-backed GPU support, `gpu`/`metal` requests fall back
to CPU.

Example:

```bash
MLX_DEFAULT_DEVICE=gpu bundle exec ruby your_script.rb
```

## Onnx/WebGPU Support

MLX Ruby exposes Graph IR/ONNX/WebGPU entrypoints on `MLX::GraphIR`.

Architecture boundary:

- Public API (`MLX::GraphIR`):
  - `export_graph_ir_json`
  - `validate!`
  - `compatibility_report`
  - `graph_ir_to_onnx_json`
  - `onnx_json_to_onnx`
  - `export_onnx_json`
  - `export_onnx_webgpu_harness`
  - `smoke_test_onnx_webgpu_harness`
- Internal implementation modules:
  - `MLX::GraphIR`
  - `MLX::GraphIR::Exporter`
  - `MLX::GraphIR::ONNX::Exporter`
  - `MLX::GraphIR::ONNX::PythonBuilder`
  - `MLX::GraphIR::WebGPUHarness`

End-to-end flow:

1. Export Graph IR with `MLX::GraphIR.export_graph_ir_json`.
2. Validate and gate conversion with `MLX::GraphIR.validate!` and
   `MLX::GraphIR.compatibility_report`.
3. Generate JSON ONNX stubs with `MLX::GraphIR.graph_ir_to_onnx_json`.
4. Export binary ONNX with `MLX::GraphIR.onnx_json_to_onnx`
   (`external_data` options are available for large models), and/or export
   ONNX JSON directly from trace with `MLX::GraphIR.export_onnx_json`.
5. Package browser harness assets with `MLX::GraphIR.export_onnx_webgpu_harness`.
6. Verify runtime behavior with `MLX::GraphIR.smoke_test_onnx_webgpu_harness`.

Harness artifacts from `export_onnx_webgpu_harness`:

- `model.onnx`
- `harness.manifest.json`
- `inputs.example.json`
- `index.html`
- `harness.js`
- optional external data file (for example `model.data`)

Smoke telemetry from `smoke_test_onnx_webgpu_harness` uses
`onnx_webgpu_telemetry_v1` and reports provider selection/fallback details
(`selected_provider`, `requested_providers`, `fallback_used`) plus timing
fields (`run_timings_ms`, `model_load_latency_ms`,
`first_inference_latency_ms`, `steady_state_inference_latency_ms`).

Operational requirements:

- `onnx_json_to_onnx` requires `python3` with the `onnx` package available.
- `onnx_json_to_onnx` requires a path-like target (not IO).
- Browser smoke tests require Node.js + Playwright (`web/`) and optionally
  local `onnxruntime-web` assets.
- Harness execution providers are `webgpu` and `wasm`.
- `web:assets` exports GPT-2 and Stable Diffusion assets each run; nanoGPT ONNX
  export is skipped unless trained nanoGPT artifacts already exist.

Demo asset workflows:

- Generate browser assets: `bundle exec rake web:assets`
- Start local demo server: `bundle exec rake web:start`

Web Demo quickstart:

```bash
bundle exec rake web:assets
bundle exec rake web:start
```

Then open:

- `http://127.0.0.1:3030/`
- `http://127.0.0.1:3030/demo/gpt2/`
- `http://127.0.0.1:3030/demo/nanogpt/`
- `http://127.0.0.1:3030/demo/stable_diffusion/`

API reference:

- `docs/src/ruby/export.rst`

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

Test task shortcuts:

- CPU-only: `bundle exec rake "test[cpu]"`
- GPU-only: `bundle exec rake "test[gpu]"`
- Installed gem artifact test: `bundle exec rake test:gem`

Strict mode (per-file timeout):

```bash
MLX_STRICT_TESTS=1 MLX_TEST_TIMEOUT=30 bundle exec rake test
```

### Benchmarks (Ruby vs Python implementations)

List tasks:

```bash
bundle exec rake -T
```

Run one benchmark lane:

```bash
bundle exec rake "benchmark:cpu[local]"
```

Run all benchmark suites:

```bash
bundle exec rake "benchmark:all[local,examples]"
```

Install benchmark Python dependencies into your active Python environment (for asdf users, this is the Python selected by your current shell / `.tool-versions`):

```bash
bundle exec rake benchmark:deps
```

Common benchmark environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DEVICE` | `gpu` | Compute device (`cpu`, `gpu`, or `metal`) |
| `RUNS` | `50` | Timed iterations (`ITERATIONS` is accepted for compatibility) |
| `WARMUP` | `10` | Warmup iterations |
| `BATCH` | `8` | Batch size |
| `SEQUENCE_LENGTH` | `128` | Source sequence length |
| `TARGET_SEQUENCE_LENGTH` | `64` | Target sequence length |
| `DIMENSIONS` | `256` | Model width |
| `HEADS` | `8` | Attention heads |
| `LAYERS` | `4` | Number of layers |
| `PYTHON` | `python3` | Python executable for cross-language comparison |
| `BENCHMARK_DEVICES` | `cpu,gpu` | Devices for top-level `rake benchmark` |
| `EXAMPLES_MODE` | `dsl` | Examples-submodule mode (`dsl` or `no_dsl`) |
| `WEBGPU_TIMEOUT` | `180` | WebGPU harness timeout seconds |
| `WEBGPU_WARMUP` | benchmark warmup | WebGPU warmup runs |
| `WEBGPU_MEASURE` | benchmark runs | WebGPU measured runs |
| `REQUIRE_WEBGPU` | unset | Fail instead of skip when WebGPU provider is unavailable |

Quick benchmark smoke command:

```bash
bundle exec rake "benchmark:cpu[local]" RUNS=5 WARMUP=1
```

### Build docs

From the repo root:

```bash
# One-time setup
brew install doxygen                    # macOS (or install doxygen via apt on Linux)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bundle install

# Generate docs
bundle exec rake docs:build
```

Docs are written to `docs/build/html`.

```bash
# Quick local preview
ruby -run -e httpd docs/build/html -p 8000
```

Then open `http://localhost:8000/`.

The repo’s Pages workflow builds docs together with the web demo for deployment.

## Repository layout

- `lib/`: Ruby API surface (`core`, `nn`, `optimizers`, `dsl`, distributed
  utilities), with Graph IR/ONNX implementation modules under
  `lib/mlx/graph_ir/**`.
- `ext/mlx/`: native extension build bridge (`extconf.rb`, C++ binding entry).
- `mlx/`: upstream MLX submodule.
- `examples/web/`: web demo model/export helpers (GPT-2, nanoGPT, Stable Diffusion).
- `tasks/`: rake task implementations (`build`, `test`, `docs`, `benchmark`,
  `web`, training/assets exporters).
- `web/`: static demo site, generated assets, ONNX WebGPU harness templates.
- `test/`: unit/task/parity suites.
- `test/parity/scripts/`: coverage/report generators.
- `docs/`: Sphinx + Doxygen documentation sources.

## Troubleshooting

- `missing MLX include dir`: initialize submodules (`git submodule update --init --recursive`).
- Native extension does not load: rebuild manually:

```bash
cd ext/mlx
ruby extconf.rb
make -j4
```

- `onnx_json_to_onnx` fails with Python import errors: ensure `onnx` is installed in
  the Python selected by `PYTHON`/`python3`.
- On Apple silicon, verify native architecture:

```bash
ruby -e 'require "rbconfig"; puts RbConfig::CONFIG["host_cpu"]'
```

- Web smoke fails due missing runtime dependencies: run
  `bundle exec rake deps:web` (installs/checks `onnx`, `node`/`npm`/`npx`,
  `playwright`, and `onnxruntime-web`).
- If CMake configure fails intermittently, rerun `ruby extconf.rb`; the build script already includes a clean-retry path.

## Contributing

- Open pull requests against this repository.
- Keep parity artifacts in `test/parity/reports/` in sync with tool/script changes.
- Follow upstream MLX contributor guidance where applicable: [mlx/CONTRIBUTING.md](https://github.com/ml-explore/mlx/blob/main/CONTRIBUTING.md).

CI currently runs on `ubuntu-22.04` and `macos-14` with Ruby `3.4` and `4.0`.

## License

`mlx` gem is distributed under the MIT license (see `LICENSE`).
