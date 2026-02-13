# frozen_string_literal: true

require_relative "test_helper"

class Phase34RollTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_roll_flattened
    x = MLX::Core.array([1, 2, 3, 4, 5], MLX::Core.int32)
    assert_equal [4, 5, 1, 2, 3], MLX::Core.roll(x, 2).to_a
  end

  def test_roll_with_single_axis
    m = MLX::Core.array([[1, 2, 3], [4, 5, 6]], MLX::Core.int32)
    assert_equal [[3, 1, 2], [6, 4, 5]], MLX::Core.roll(m, 1, 1).to_a
  end

  def test_roll_with_multi_axis_shifts
    m = MLX::Core.array([[1, 2, 3], [4, 5, 6]], MLX::Core.int32)
    assert_equal [[5, 6, 4], [2, 3, 1]], MLX::Core.roll(m, [1, -1], [0, 1]).to_a
  end
end
