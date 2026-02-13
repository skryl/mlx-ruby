# frozen_string_literal: true

require_relative "test_helper"

class Phase179GaussianNllLossTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_gaussian_nll_loss_values
    inputs = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    targets = MLX::Core.array([1.0, 4.0], MLX::Core.float32)
    vars = MLX::Core.array([1.0, 4.0], MLX::Core.float32)

    loss_mean = MLX::NN.gaussian_nll_loss(inputs, targets, vars, full: false, reduction: "mean")
    loss_full = MLX::NN.gaussian_nll_loss(inputs, targets, vars, full: true, reduction: "mean")

    assert_in_delta 0.5965735, loss_mean.item, 1e-5
    assert_in_delta 1.515512, loss_full.item, 1e-5
  end

  def test_gaussian_nll_loss_shape_validation
    inputs = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    targets = MLX::Core.array([1.0], MLX::Core.float32)
    vars = MLX::Core.array([1.0, 2.0], MLX::Core.float32)

    assert_raises(ArgumentError) do
      MLX::NN.gaussian_nll_loss(inputs, targets, vars)
    end

    assert_raises(ArgumentError) do
      MLX::NN.gaussian_nll_loss(inputs, MLX::Core.array([1.0, 2.0], MLX::Core.float32), MLX::Core.array([1.0], MLX::Core.float32))
    end
  end
end
