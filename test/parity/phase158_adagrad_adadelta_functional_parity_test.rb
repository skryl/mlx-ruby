# frozen_string_literal: true

require_relative "test_helper"

class Phase158AdagradAdadeltaFunctionalParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_adagrad_updates_and_state
    opt = MLX::Optimizers::Adagrad.new(learning_rate: 0.1, eps: 1e-8)

    params = { "w" => MLX::Core.array([10.0], MLX::Core.float32) }
    grads = { "w" => MLX::Core.array([2.0], MLX::Core.float32) }

    out1 = opt.apply_gradients(grads, params)
    out2 = opt.apply_gradients(grads, out1)

    assert_nested_close [9.9], out1.fetch("w").to_a, 1e-4
    assert_nested_close [9.82929], out2.fetch("w").to_a, 1e-4
    assert_nested_close [8.0], opt.state.fetch("w").fetch("v").to_a, 1e-4
  end

  def test_adadelta_updates_and_state
    opt = MLX::Optimizers::AdaDelta.new(learning_rate: 1.0, rho: 0.9, eps: 1e-6)

    params = { "w" => MLX::Core.array([10.0], MLX::Core.float32) }
    grads = { "w" => MLX::Core.array([2.0], MLX::Core.float32) }

    out = opt.apply_gradients(grads, params)

    assert_nested_close [9.99684], out.fetch("w").to_a, 1e-4
    assert_nested_close [0.4], opt.state.fetch("w").fetch("v").to_a, 1e-4
    assert_nested_close [1.0e-6], opt.state.fetch("w").fetch("u").to_a, 1e-7
  end

  def test_adagrad_and_adadelta_validate_params
    assert_raises(ArgumentError) { MLX::Optimizers::Adagrad.new(learning_rate: 0.1, eps: -1e-8) }
    assert_raises(ArgumentError) { MLX::Optimizers::AdaDelta.new(learning_rate: 0.1, rho: -0.1) }
    assert_raises(ArgumentError) { MLX::Optimizers::AdaDelta.new(learning_rate: 0.1, eps: -1e-6) }
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
