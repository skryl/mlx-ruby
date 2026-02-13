# frozen_string_literal: true

require_relative "test_helper"

class Phase15AxisManipulationTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_moveaxis_matches_transpose_equivalent
    x = MLX::Core.reshape(MLX::Core.arange(24, MLX::Core.int32), [2, 3, 4])

    moved = MLX::Core.moveaxis(x, 0, 2)
    transposed = MLX::Core.transpose(x, [1, 2, 0])

    assert_equal [3, 4, 2], moved.shape
    assert_equal transposed.to_a, moved.to_a
  end

  def test_swapaxes_matches_transpose_equivalent
    x = MLX::Core.reshape(MLX::Core.arange(24, MLX::Core.int32), [2, 3, 4])

    swapped = MLX::Core.swapaxes(x, 0, 1)
    transposed = MLX::Core.transpose(x, [1, 0, 2])

    assert_equal [3, 2, 4], swapped.shape
    assert_equal transposed.to_a, swapped.to_a
  end
end
