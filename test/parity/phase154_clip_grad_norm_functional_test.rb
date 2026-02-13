# frozen_string_literal: true

require_relative "test_helper"

class Phase154ClipGradNormFunctionalTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_clip_grad_norm_scales_by_global_norm
    grads = {
      "w1" => MLX::Core.array([3.0, 4.0], MLX::Core.float32),
      "w2" => MLX::Core.array([0.0], MLX::Core.float32)
    }

    clipped, total_norm = MLX::Optimizers.clip_grad_norm(grads, 2.0)

    assert_in_delta 5.0, scalar(total_norm), 1e-4
    assert_nested_close [1.2, 1.6], clipped.fetch("w1").to_a, 1e-3
    assert_nested_close [0.0], clipped.fetch("w2").to_a, 1e-6
  end

  def test_clip_grad_norm_keeps_small_gradients_unchanged
    grads = { "w" => MLX::Core.array([0.1, 0.2], MLX::Core.float32) }

    clipped, total_norm = MLX::Optimizers.clip_grad_norm(grads, 10.0)

    assert_operator scalar(total_norm), :<, 10.0
    assert_nested_close [0.1, 0.2], clipped.fetch("w").to_a, 1e-6
  end

  private

  def scalar(value)
    value.respond_to?(:item) ? value.item : value
  end

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
