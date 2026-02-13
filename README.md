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

## Examples

The MLX examples ecosystem contains Ruby-parity artifacts and integration
references in this repository under `ruby/tools` and `ruby/test`.

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
gem "mlx", path: "/path/to/codex-ruby/ruby"
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
