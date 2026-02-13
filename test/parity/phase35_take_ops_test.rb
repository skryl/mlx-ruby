# frozen_string_literal: true

require_relative "test_helper"

class Phase35TakeOpsTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_take_flattened_and_axis
    x = MLX::Core.array([[10, 11, 12], [20, 21, 22]], MLX::Core.int32)

    assert_equal 20, MLX::Core.take(x, 3).to_a
    assert_equal [[12, 10], [22, 20]], MLX::Core.take(x, [2, 0], 1).to_a
    assert_equal [21, 11], MLX::Core.take(x, [4, 1]).to_a
  end

  def test_take_along_axis
    a = MLX::Core.array([[10, 30, 20], [40, 60, 50]], MLX::Core.int32)

    idx = MLX::Core.array([[2, 0], [1, 2]], MLX::Core.int32)
    assert_equal [[20, 10], [60, 50]], MLX::Core.take_along_axis(a, idx, 1).to_a

    flat_idx = MLX::Core.array([5, 0, 3], MLX::Core.int32)
    assert_equal [50, 10, 40], MLX::Core.take_along_axis(a, flat_idx).to_a
  end
end
