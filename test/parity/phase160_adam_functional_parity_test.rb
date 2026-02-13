# frozen_string_literal: true

require_relative "test_helper"

class Phase160AdamFunctionalParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_adam_updates_and_state_without_bias_correction
    opt = MLX::Optimizers::Adam.new(
      learning_rate: 0.1,
      betas: [0.9, 0.999],
      eps: 1e-8,
      bias_correction: false
    )

    params = { "w" => MLX::Core.array([10.0], MLX::Core.float32) }
    grads = { "w" => MLX::Core.array([2.0], MLX::Core.float32) }

    out1 = opt.apply_gradients(grads, params)
    out2 = opt.apply_gradients(grads, out1)

    assert_nested_close [9.68377], out1.fetch("w").to_a, 1e-4
    assert_nested_close [9.25885], out2.fetch("w").to_a, 1e-4
    assert_nested_close [0.38], opt.state.fetch("w").fetch("m").to_a, 1e-4
    assert_nested_close [0.007996], opt.state.fetch("w").fetch("v").to_a, 1e-6
  end

  def test_adam_bias_correction_uses_incremented_step
    opt = MLX::Optimizers::Adam.new(
      learning_rate: 0.1,
      betas: [0.9, 0.999],
      eps: 1e-8,
      bias_correction: true
    )

    params = { "w" => MLX::Core.array([10.0], MLX::Core.float32) }
    grads = { "w" => MLX::Core.array([2.0], MLX::Core.float32) }

    out = opt.apply_gradients(grads, params)

    assert_equal 1, opt.step
    assert_nested_close [9.9], out.fetch("w").to_a, 1e-4
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
