# frozen_string_literal: true

require_relative "test_helper"

class Phase39ViewLikeOpsTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_real_imag_conjugate_and_stop_gradient
    x = MLX::Core.array([1.0, -2.0, 3.5], MLX::Core.float32)

    assert_nested_close [1.0, -2.0, 3.5], MLX::Core.real(x).to_a
    assert_nested_close [0.0, 0.0, 0.0], MLX::Core.imag(x).to_a
    assert_nested_close [1.0, -2.0, 3.5], MLX::Core.conjugate(x).to_a
    assert_nested_close [1.0, -2.0, 3.5], MLX::Core.conj(x).to_a
    assert_nested_close [1.0, -2.0, 3.5], MLX::Core.stop_gradient(x).to_a
  end

  def test_contiguous_preserves_values
    x = MLX::Core.reshape(MLX::Core.arange(12, MLX::Core.float32), [3, 4])
    assert_nested_close x.to_a, MLX::Core.contiguous(x).to_a
  end

  private

  def assert_nested_close(expected, actual, atol = 1e-5)
    assert_equal structure_signature(expected), structure_signature(actual)
    flatten(expected).zip(flatten(actual)).each do |exp, got|
      assert_in_delta exp, got, atol
    end
  end

  def flatten(value)
    return [value] unless value.is_a?(Array)

    value.flat_map { |x| flatten(x) }
  end

  def structure_signature(value)
    return :scalar unless value.is_a?(Array)

    [value.length, *(value.map { |v| structure_signature(v) })]
  end
end
