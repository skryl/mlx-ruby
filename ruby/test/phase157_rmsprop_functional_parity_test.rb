# frozen_string_literal: true

require_relative "test_helper"

class Phase157RmspropFunctionalParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_rmsprop_step_updates_and_state
    opt = MLX::Optimizers::RMSprop.new(learning_rate: 0.1, alpha: 0.99, eps: 1e-8)

    params = { "w" => MLX::Core.array([10.0], MLX::Core.float32) }
    grads = { "w" => MLX::Core.array([2.0], MLX::Core.float32) }

    out1 = opt.apply_gradients(grads, params)
    out2 = opt.apply_gradients(grads, out1)

    assert_nested_close [9.0], out1.fetch("w").to_a, 1e-4
    assert_nested_close [8.2911], out2.fetch("w").to_a, 1e-3
    assert_nested_close [0.0796], opt.state.fetch("w").fetch("v").to_a, 1e-4
  end

  def test_rmsprop_validates_alpha_and_eps
    assert_raises(ArgumentError) do
      MLX::Optimizers::RMSprop.new(learning_rate: 0.1, alpha: -0.1)
    end

    assert_raises(ArgumentError) do
      MLX::Optimizers::RMSprop.new(learning_rate: 0.1, eps: -1e-8)
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
