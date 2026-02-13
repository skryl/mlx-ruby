# frozen_string_literal: true

require_relative "test_helper"

class Phase9LinalgBroadcastTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_matmul_and_default_transpose
    a = MLX::Core.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], MLX::Core.float32)
    b = MLX::Core.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], MLX::Core.float32)

    out = MLX::Core.matmul(a, b)
    assert_equal [2, 2], out.shape
    assert_nested_close [[58.0, 64.0], [139.0, 154.0]], out.to_a

    at = MLX::Core.transpose(a)
    assert_equal [3, 2], at.shape
    assert_nested_close [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], at.to_a
  end

  def test_broadcast_to_and_broadcast_arrays
    x = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)
    y = MLX::Core.broadcast_to(x, [2, 3])

    assert_equal [2, 3], y.shape
    assert_nested_close [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], y.to_a

    left = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)
    right = MLX::Core.array([[10.0], [20.0]], MLX::Core.float32)

    expanded = MLX::Core.broadcast_arrays([left, right])
    assert_equal 2, expanded.length
    assert_equal [2, 3], expanded[0].shape
    assert_equal [2, 3], expanded[1].shape
    assert_nested_close [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], expanded[0].to_a
    assert_nested_close [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]], expanded[1].to_a
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
