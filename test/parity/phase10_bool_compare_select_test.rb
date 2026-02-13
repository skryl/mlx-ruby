# frozen_string_literal: true

require_relative "test_helper"

class Phase10BoolCompareSelectTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_elementwise_comparisons
    a = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)
    b = MLX::Core.array([1.0, 1.5, 4.0], MLX::Core.float32)

    eq = MLX::Core.equal(a, b)
    neq = MLX::Core.not_equal(a, b)
    gt = MLX::Core.greater(a, b)
    lt = MLX::Core.less(a, b)

    assert_equal :bool_, eq.dtype.name
    assert_equal [true, false, false], eq.to_a
    assert_equal [false, true, true], neq.to_a
    assert_equal [false, true, false], gt.to_a
    assert_equal [false, false, true], lt.to_a
  end

  def test_where_selects_values_and_supports_broadcast
    mask = MLX::Core.array([[true, false, true], [false, true, false]], MLX::Core.bool_)
    left = MLX::Core.array([10.0, 20.0, 30.0], MLX::Core.float32)
    right = MLX::Core.array([[1.0], [2.0]], MLX::Core.float32)

    out = MLX::Core.where(mask, left, right)

    assert_equal [2, 3], out.shape
    assert_nested_close [[10.0, 1.0, 30.0], [2.0, 20.0, 2.0]], out.to_a
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
