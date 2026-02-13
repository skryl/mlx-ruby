# frozen_string_literal: true

require_relative "test_helper"

class Phase26ArithmeticFunctionOpsTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_module_level_binary_arithmetic_ops
    a = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)
    b = MLX::Core.array([4.0, 5.0, 6.0], MLX::Core.float32)

    assert_nested_close [5.0, 7.0, 9.0], MLX::Core.add(a, b).to_a
    assert_nested_close [-3.0, -3.0, -3.0], MLX::Core.subtract(a, b).to_a
    assert_nested_close [4.0, 10.0, 18.0], MLX::Core.multiply(a, b).to_a
    assert_nested_close [0.25, 0.4, 0.5], MLX::Core.divide(a, b).to_a
  end

  def test_module_level_power_and_remainder_with_scalar
    a = MLX::Core.array([1, 2, 3], MLX::Core.int32)
    b = MLX::Core.array([4, 5, 6], MLX::Core.int32)

    assert_equal [1, 4, 9], MLX::Core.power(a, 2).to_a
    assert_equal [0, 1, 2], MLX::Core.remainder(b, 4).to_a
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
