# frozen_string_literal: true

require_relative "test_helper"

class Phase108PackageSkeletonTest < Minitest::Test
  def test_ruby_package_tree_contains_python_mlx_equivalent_files
    expected = %w[
      lib/mlx/nn/base.rb
      lib/mlx/nn/init.rb
      lib/mlx/nn/losses.rb
      lib/mlx/nn/utils.rb
      lib/mlx/nn/layers.rb
      lib/mlx/nn/layers/activations.rb
      lib/mlx/nn/layers/base.rb
      lib/mlx/nn/layers/containers.rb
      lib/mlx/nn/layers/convolution.rb
      lib/mlx/nn/layers/convolution_transpose.rb
      lib/mlx/nn/layers/distributed.rb
      lib/mlx/nn/layers/dropout.rb
      lib/mlx/nn/layers/embedding.rb
      lib/mlx/nn/layers/linear.rb
      lib/mlx/nn/layers/normalization.rb
      lib/mlx/nn/layers/pooling.rb
      lib/mlx/nn/layers/positional_encoding.rb
      lib/mlx/nn/layers/quantized.rb
      lib/mlx/nn/layers/recurrent.rb
      lib/mlx/nn/layers/transformer.rb
      lib/mlx/nn/layers/upsample.rb
      lib/mlx/optimizers/optimizers.rb
      lib/mlx/optimizers/schedulers.rb
      lib/mlx/distributed_utils/common.rb
      lib/mlx/distributed_utils/config.rb
      lib/mlx/distributed_utils/launch.rb
    ]

    expected.each do |rel|
      abs = File.join(RUBY_ROOT, rel)
      assert File.exist?(abs), "missing skeleton file: #{rel}"
    end
  end
end
