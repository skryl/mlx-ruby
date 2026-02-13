# frozen_string_literal: true

require_relative "test_helper"

class Phase19LogicalOpsTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_logical_unary_and_binary_ops
    a = MLX::Core.array([true, false, true], MLX::Core.bool_)
    b = MLX::Core.array([false, false, true], MLX::Core.bool_)

    assert_equal [false, true, false], MLX::Core.logical_not(a).to_a
    assert_equal [false, false, true], MLX::Core.logical_and(a, b).to_a
    assert_equal [true, false, true], MLX::Core.logical_or(a, b).to_a
  end

  def test_logical_ops_support_scalar_inputs
    a = MLX::Core.array([true, false, true], MLX::Core.bool_)

    assert_equal [true, false, true], MLX::Core.logical_and(a, true).to_a
    assert_equal [true, true, true], MLX::Core.logical_or(a, true).to_a
  end
end
