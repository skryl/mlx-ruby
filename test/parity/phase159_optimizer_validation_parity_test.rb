# frozen_string_literal: true

require_relative "test_helper"

class Phase159OptimizerValidationParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_optimizer_constructor_validations
    assert_raises(ArgumentError) do
      MLX::Optimizers::SGD.new(learning_rate: 0.1, nesterov: true)
    end

    assert_raises(ArgumentError) do
      MLX::Optimizers::SGD.new(
        learning_rate: 0.1,
        momentum: 0.9,
        dampening: 0.1,
        nesterov: true
      )
    end

    assert_raises(ArgumentError) { MLX::Optimizers::RMSprop.new(learning_rate: 0.1, alpha: -0.1) }
    assert_raises(ArgumentError) { MLX::Optimizers::RMSprop.new(learning_rate: 0.1, eps: -1e-8) }
    assert_raises(ArgumentError) { MLX::Optimizers::Adagrad.new(learning_rate: 0.1, eps: -1e-8) }
    assert_raises(ArgumentError) { MLX::Optimizers::AdaDelta.new(learning_rate: 0.1, rho: -0.1) }
    assert_raises(ArgumentError) { MLX::Optimizers::AdaDelta.new(learning_rate: 0.1, eps: -1e-6) }
    assert_raises(ArgumentError) { MLX::Optimizers::Adamax.new(learning_rate: 0.1, eps: -1e-8) }
  end
end
