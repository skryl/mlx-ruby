# frozen_string_literal: true

require_relative "test_helper"

class Phase156SgdFunctionalParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_sgd_without_momentum
    opt = MLX::Optimizers::SGD.new(learning_rate: 0.1)

    params = { "w" => MLX::Core.array([10.0], MLX::Core.float32) }
    grads = { "w" => MLX::Core.array([2.0], MLX::Core.float32) }

    out = opt.apply_gradients(grads, params)

    assert_nested_close [9.8], out.fetch("w").to_a
  end

  def test_sgd_with_momentum_accumulates_velocity
    opt = MLX::Optimizers::SGD.new(learning_rate: 0.1, momentum: 0.9)
    params = { "w" => MLX::Core.array([10.0], MLX::Core.float32) }
    grads = { "w" => MLX::Core.array([1.0], MLX::Core.float32) }

    out1 = opt.apply_gradients(grads, params)
    out2 = opt.apply_gradients(grads, out1)

    assert_nested_close [9.9], out1.fetch("w").to_a
    assert_nested_close [9.71], out2.fetch("w").to_a, 1e-4
    assert_nested_close [1.9], opt.state.fetch("w").fetch("v").to_a, 1e-4
  end

  def test_sgd_weight_decay
    opt = MLX::Optimizers::SGD.new(learning_rate: 0.1, weight_decay: 0.1)

    params = { "w" => MLX::Core.array([10.0], MLX::Core.float32) }
    grads = { "w" => MLX::Core.array([1.0], MLX::Core.float32) }

    out = opt.apply_gradients(grads, params)

    assert_nested_close [9.8], out.fetch("w").to_a
  end

  def test_sgd_nesterov_requires_positive_momentum_and_zero_dampening
    assert_raises(ArgumentError) do
      MLX::Optimizers::SGD.new(learning_rate: 0.1, nesterov: true)
    end

    assert_raises(ArgumentError) do
      MLX::Optimizers::SGD.new(learning_rate: 0.1, momentum: 0.9, dampening: 0.1, nesterov: true)
    end
  end

  private

  def assert_nested_close(expected, actual, atol = 1e-5)
    assert_equal shape_signature(expected), shape_signature(actual)
    flatten(expected).zip(flatten(actual)).each do |exp, got|
      assert_in_delta exp, got, atol
    end
  end

  def flatten(value)
    return [value] unless value.is_a?(Array)

    value.flat_map { |item| flatten(item) }
  end

  def shape_signature(value)
    return :scalar unless value.is_a?(Array)

    [value.length, *(value.map { |item| shape_signature(item) })]
  end
end
