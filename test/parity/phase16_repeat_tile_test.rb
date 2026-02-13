# frozen_string_literal: true

require_relative "test_helper"

class Phase16RepeatTileTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_repeat_with_and_without_axis
    x = MLX::Core.array([[1, 2], [3, 4]], MLX::Core.int32)

    assert_equal [1, 1, 2, 2, 3, 3, 4, 4], MLX::Core.repeat(x, 2).to_a
    assert_equal [[1, 1, 2, 2], [3, 3, 4, 4]], MLX::Core.repeat(x, 2, 1).to_a
  end

  def test_tile_with_scalar_and_vector_reps
    v = MLX::Core.array([1, 2], MLX::Core.int32)
    m = MLX::Core.array([[1, 2], [3, 4]], MLX::Core.int32)

    assert_equal [1, 2, 1, 2, 1, 2], MLX::Core.tile(v, 3).to_a
    assert_equal [[1, 2], [3, 4], [1, 2], [3, 4]], MLX::Core.tile(m, [2, 1]).to_a
  end
end
