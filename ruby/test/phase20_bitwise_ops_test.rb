# frozen_string_literal: true

require_relative "test_helper"

class Phase20BitwiseOpsTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_bitwise_binary_ops
    a = MLX::Core.array([1, 2, 3], MLX::Core.int32)
    b = MLX::Core.array([3, 1, 1], MLX::Core.int32)

    assert_equal [1, 0, 1], MLX::Core.bitwise_and(a, b).to_a
    assert_equal [3, 3, 3], MLX::Core.bitwise_or(a, b).to_a
    assert_equal [2, 3, 2], MLX::Core.bitwise_xor(a, b).to_a
  end

  def test_bitwise_supports_scalar_and_invert
    a = MLX::Core.array([1, 2, 3], MLX::Core.int32)

    assert_equal [1, 0, 1], MLX::Core.bitwise_and(a, 1).to_a
    assert_equal [-1, -2, -3], MLX::Core.bitwise_invert(MLX::Core.array([0, 1, 2], MLX::Core.int32)).to_a
  end
end
