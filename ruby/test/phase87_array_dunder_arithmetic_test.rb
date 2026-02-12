# frozen_string_literal: true

require_relative "test_helper"

class Phase87ArrayDunderArithmeticTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_right_hand_and_unary_dunders
    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)

    assert_equal [3.0, 4.0], x.__radd__(2.0).to_a
    assert_equal [1.0, 0.0], x.__rsub__(2.0).to_a
    assert_equal [2.0, 4.0], x.__rmul__(2.0).to_a
    assert_equal [2.0, 1.0], x.__rtruediv__(2.0).to_a
    assert_equal [1.0, 4.0], x.__pow__(2.0).to_a
    assert_equal [2.0, 4.0], x.__rpow__(2.0).to_a
    assert_equal [-1.0, -2.0], x.__neg__.to_a
    assert_equal [1.0, 2.0], x.__abs__.to_a
  end

  def test_integer_dunders_for_bitwise_and_shift
    x = MLX::Core.array([1, 3, 7], MLX::Core.int32)
    y = MLX::Core.array([6, 5, 1], MLX::Core.int32)

    assert_equal [0, 1, 1], x.__and__(y).to_a
    assert_equal [7, 7, 7], x.__or__(y).to_a
    assert_equal [7, 6, 6], x.__xor__(y).to_a
    assert_equal [-2, -4, -8], x.__invert__.to_a
    assert_equal [2, 6, 14], x.__lshift__(1).to_a
    assert_equal [0, 1, 3], x.__rshift__(1).to_a
    assert_equal [0, 1, 3], x.__floordiv__(2).to_a
    assert_equal [1, 1, 1], x.__mod__(2).to_a
    assert_equal [0, 2, 2], x.__rmod__(2).to_a
  end

  def test_inplace_dunders_return_arrays
    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    assert_equal [2.0, 3.0], x.__iadd__(1.0).to_a
    assert_equal [0.0, 1.0], x.__isub__(1.0).to_a
    assert_equal [2.0, 4.0], x.__imul__(2.0).to_a
    assert_equal [0.5, 1.0], x.__itruediv__(2.0).to_a
  end
end
