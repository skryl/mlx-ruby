# frozen_string_literal: true

require_relative "test_helper"

class Phase161AdamwAdamaxFunctionalParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_adamw_applies_decoupled_weight_decay
    opt = MLX::Optimizers::AdamW.new(
      learning_rate: 0.1,
      betas: [0.9, 0.999],
      eps: 1e-8,
      weight_decay: 0.1,
      bias_correction: false
    )

    params = { "w" => MLX::Core.array([10.0], MLX::Core.float32) }
    grads = { "w" => MLX::Core.array([0.0], MLX::Core.float32) }

    out = opt.apply_gradients(grads, params)

    assert_nested_close [9.9], out.fetch("w").to_a, 1e-4
  end

  def test_adamax_updates_and_state
    opt = MLX::Optimizers::Adamax.new(learning_rate: 0.1, betas: [0.9, 0.999], eps: 1e-8)

    params = { "w" => MLX::Core.array([10.0], MLX::Core.float32) }
    grads = { "w" => MLX::Core.array([2.0], MLX::Core.float32) }

    out1 = opt.apply_gradients(grads, params)
    out2 = opt.apply_gradients(grads, out1)

    assert_nested_close [9.99], out1.fetch("w").to_a, 1e-5
    assert_nested_close [9.971], out2.fetch("w").to_a, 1e-4
    assert_nested_close [0.38], opt.state.fetch("w").fetch("m").to_a, 1e-4
    assert_nested_close [2.0], opt.state.fetch("w").fetch("v").to_a, 1e-4
  end

  def test_adamax_validates_eps
    assert_raises(ArgumentError) do
      MLX::Optimizers::Adamax.new(learning_rate: 0.1, eps: -1e-8)
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
