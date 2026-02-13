# frozen_string_literal: true

require_relative "test_helper"

class Phase29LogRoundDivShiftTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_log2_log10_and_round
    x = MLX::Core.array([1.0, 2.0, 4.0, 8.0], MLX::Core.float32)
    y = MLX::Core.array([1.0, 10.0, 100.0], MLX::Core.float32)

    assert_nested_close [0.0, 1.0, 2.0, 3.0], MLX::Core.log2(x).to_a, 1e-4
    assert_nested_close [0.0, 1.0, 2.0], MLX::Core.log10(y).to_a, 1e-4

    rounded = MLX::Core.round(MLX::Core.array([1.1, 1.9, -1.1, -1.9], MLX::Core.float32))
    assert_nested_close [1.0, 2.0, -1.0, -2.0], rounded.to_a

    rounded_2 = MLX::Core.round(MLX::Core.array([1.234, -1.234], MLX::Core.float32), 2)
    assert_nested_close [1.23, -1.23], rounded_2.to_a, 1e-4
  end

  def test_floor_divide_and_bit_shifts
    a = MLX::Core.array([7, 8, 9], MLX::Core.int32)

    assert_equal [3, 4, 4], MLX::Core.floor_divide(a, 2).to_a
    assert_equal [2, 4, 6], MLX::Core.left_shift(MLX::Core.array([1, 2, 3], MLX::Core.int32), 1).to_a
    assert_equal [1, 2, 4], MLX::Core.right_shift(MLX::Core.array([4, 8, 16], MLX::Core.int32), 2).to_a
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
