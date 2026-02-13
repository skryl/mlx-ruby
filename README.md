# MLX Ruby

[**Quickstart**](#quickstart) | [**Installation**](#installation) |
[**Documentation**](https://skryl.github.io/mlx-ruby/build/html/index.html) |
[**Contributing**](#contributing)

MLX Ruby is an array framework for machine learning on Apple silicon,
brought to you by Apple machine learning research.

Some key features:

- **Familiar APIs**: Ruby APIs mirror the Python MLX surface (`MLX::Core`, `MLX::NN`, `MLX::Optimizers`) and map to similar concepts for easy migration and cross-language parity work.
- **Composable transformations**: Ruby supports the same transformation-oriented workflows exposed by MLX core types.
- **Lazy execution**: Array operations are lazy until materialization is required.
- **Dynamic graph construction**: Graphs are built dynamically with simple, inspectable Ruby flows.
- **Multi-device**: Operations run on supported devices (CPU and GPU when available).
- **Unified memory model**: Arrays can be used across supported devices without manual transfer boilerplate.

MLX Ruby is designed for researchers and builders who want a simple, Ruby-native
entry point into MLX with low-level extension control when needed.

## Repository overview

This repo provides a Ruby binding for Apple MLX with the same public structure you
expect from MLX in Ruby:

- Core tensor ops in `MLX::Core`.
- Neural-network modules in `MLX::NN`.
- Optimizers, scheduling, and training helpers in `MLX::Optimizers`.
- Native extension glue in `ext/` and `lib/mlx/`.

The project is split into two main areas:

- `lib/`: Ruby API implementation and compatibility layer.
- `ext/`: C++ and CMake build files for compiling the native extension.
- `ruby/`: Bundler/test scaffolding for development workflows.
- `test/`: Ruby parity tests that validate parity coverage and behavior.
- `docs/`: Sphinx docs for public API and examples.

The binding is designed to:

- Keep MLX semantics close to Python MLX for easier porting of examples.
- Support parity testing between Ruby and upstream MLX behaviors.
- Offer a single installable gem artifact once native binaries are built.

If you are coming from MLX Python, the workflow is intentionally familiar:
build the extension, load the gem locally, run the test matrix, then move to your
own scripts.

## Examples

The `## Examples` section below shows practical Ruby examples you can paste into a
small script and run once the gem is installed.

Small transformer example (inference + one training step):

```ruby
require "mlx"

# A compact encoder-style transformer for sequence classification-style output.
class MiniTransformer < MLX::NN::Module
  def initialize(vocab_size:, seq_len:, embed_dim: 64, num_heads: 4, num_layers: 2, num_classes: 10)
    super()

    # Token and positional embeddings.
    @token_embedding = MLX::NN::Embedding.new(vocab_size, embed_dim)
    @pos_embedding = MLX::NN::Embedding.new(seq_len, embed_dim)

    # Stacked transformer encoder blocks.
    @blocks = Array.new(num_layers) do
      MLX::NN::TransformerEncoderLayer.new(
        embed_dim,
        num_heads,
        mlp_dims: embed_dim * 4,
        dropout: 0.1,
        norm_first: true
      )
    end

    # Final projection head.
    @norm = MLX::NN::LayerNorm.new(embed_dim)
    @proj = MLX::NN::Linear.new(embed_dim, num_classes)

    # Causal mask to prevent attention from seeing future tokens.
    @causal_mask = MLX::NN::MultiHeadAttention.create_additive_causal_mask(seq_len)
  end

  def call(input_ids)
    # [batch, seq_len] integer token ids -> token+position embeddings.
    positions = MLX::Core.arange(0, input_ids.shape[1], 1, MLX::Core.int32)
    x = MLX::Core.add(@token_embedding.call(input_ids), @pos_embedding.call(positions))

    # Apply encoder layers with the causal attention mask.
    @blocks.each { |block| x = block.call(x, @causal_mask) }

    # Normalize and project to class logits for every token.
    x = @norm.call(x)
    @proj.call(x)
  end
end

mx = MLX::Core
batch_size = 2
seq_len = 8
vocab_size = 100

# Synthetic IDs for a tiny forward/training pass.
input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8], [7, 6, 5, 4, 3, 2, 1, 0]], mx.int32)
targets = mx.array([[1, 2, 3, 4, 5, 6, 7, 8], [7, 6, 5, 4, 3, 2, 1, 0]], mx.int32)

model = MiniTransformer.new(vocab_size: vocab_size, seq_len: seq_len, num_classes: vocab_size)
logits = model.call(input_ids)
puts "logits shape: #{logits.shape}"

# Build loss and gradient function for a single training step.
value_and_grad = MLX::NN.value_and_grad(
  model,
  lambda do |ids, labels|
    pred = model.call(ids)

    # Flatten sequence dimension for cross-entropy.
    pred2d = MLX::Core.reshape(pred, [batch_size * seq_len, vocab_size])
    labels2d = MLX::Core.reshape(labels, [batch_size * seq_len])
    MLX::NN.cross_entropy(pred2d, labels2d, reduction: "mean")
  end
)

# Compute gradients, run one optimization step, and materialize the scalar loss.
loss, grads = value_and_grad.call(input_ids, targets)
optimizer = MLX::Optimizers::AdamW.new(learning_rate: 1e-3)
optimizer.update(model, grads)
mx.eval(loss)
puts "training loss: #{loss.item()}"
```

Small CNN example (forward + one training step):

```ruby
require "mlx"

class MiniCnnClassifier < MLX::NN::Module
  def initialize(num_classes: 10)
    super()

    @conv1 = MLX::NN::Conv2d.new(1, 16, 3, padding: 1)
    @conv2 = MLX::NN::Conv2d.new(16, 32, 3, padding: 1)
    @relu = MLX::NN::ReLU.new
    @pool = MLX::NN::MaxPool2d.new(2, stride: 2)
    @fc1 = MLX::NN::Linear.new(32 * 7 * 7, 64)
    @fc2 = MLX::NN::Linear.new(64, num_classes)
  end

  def call(images)
    x = @conv1.call(images)
    x = @relu.call(x)
    x = @pool.call(x)

    x = @conv2.call(x)
    x = @relu.call(x)
    x = @pool.call(x)

    batch_size = x.shape[0]
    x = MLX::Core.reshape(x, [batch_size, 32 * 7 * 7])
    x = @fc1.call(x)
    x = @relu.call(x)
    @fc2.call(x)
  end
end

mx = MLX::Core
batch_size = 4
num_classes = 10

# Synthetic image batch [batch, height, width, channels], matching MNIST-like layout.
images = MLX::Core.random_uniform([batch_size, 28, 28, 1], 0.0, 1.0, MLX::Core.float32)
labels = MLX::Core.array([1, 3, 4, 7], MLX::Core.int32)

cnn = MiniCnnClassifier.new(num_classes: num_classes)
logits = cnn.call(images)
puts "logits shape: #{logits.shape}"

loss_and_grad = MLX::NN.value_and_grad(
  cnn,
  lambda do |x, y|
    preds = cnn.call(x)
    MLX::NN.cross_entropy(preds, y, reduction: "mean")
  end
)

loss, grads = loss_and_grad.call(images, labels)
optimizer = MLX::Optimizers::Adam.new(learning_rate: 1e-3)
optimizer.update(cnn, grads)
mx.eval(loss)
puts "training loss: #{loss.item()}"
```

## Quickstart

Require the gem and run a tiny operation:

```ruby
require "mlx"

x = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)
y = x + 1
puts y.to_a
```

Train a small module:

```ruby
class ToyModel < MLX::NN::Module
  def initialize
    @linear = MLX::NN::Linear.new(3, 1)
  end

  def call(x)
    @linear.call(x)
  end
end
```

## Installation

The Ruby binding is built as a native extension, so you'll typically run it from
the repository or via a built gem artifact.

### From source (recommended in development)

```bash
cd ruby
bundle install
bundle exec rake test
```

To build the gem:

```bash
cd ruby
gem build mlx.gemspec
```

That produces `mlx-<version>.gem`, which you can install locally:

```bash
gem install ./mlx-<version>.gem
```

### In a project Gemfile

```ruby
gem "mlx", path: "../mlx-ruby" # from another Gemfile in the same workspace
```

## Development

Run the full Ruby test suite:

```bash
cd ruby
bundle exec rake test
```

Common commands used in this directory:

```bash
bundle exec rake          # default: test
bundle exec rake test      # run all tests under ruby/test
```

## Contributing

See the main MLX contribution guidelines for repository-wide process:
[CONTRIBUTING.md](https://github.com/skryl/mlx-ruby/tree/main/CONTRIBUTING.md).
For Ruby-specific parity and tooling updates, use the `ruby/` directory and keep
`ruby/tools/parity` and `ruby/tools/parity/reports` aligned with generated
artifacts.

## Citing MLX

The MLX software suite was initially developed with equal contribution by Awni
Hannun, Jagrit Digani, Angelos Katharopoulos, and Ronan Collobert. If you
find MLX useful in your research and wish to cite it, please use the following
BibTeX entry:

```text
@software{mlx2023,
  author = {Awni Hannun and Jagrit Digani and Angelos Katharopoulos and Ronan Collobert},
  title = {{MLX}: Efficient and flexible machine learning on Apple silicon},
  url = {https://github.com/ml-explore},
  version = {0.0},
  year = {2023},
}
```
