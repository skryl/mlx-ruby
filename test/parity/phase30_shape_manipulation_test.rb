# frozen_string_literal: true

require_relative "test_helper"

class Phase30ShapeManipulationTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_flatten_squeeze_and_expand_dims
    x = MLX::Core.reshape(MLX::Core.arange(1, 7, 1, MLX::Core.int32), [2, 3])
    assert_equal [1, 2, 3, 4, 5, 6], MLX::Core.flatten(x).to_a

    y = MLX::Core.array([[[1], [2], [3]], [[4], [5], [6]]], MLX::Core.int32)
    assert_equal [2, 3], MLX::Core.squeeze(y).shape
    assert_equal [[1, 2, 3], [4, 5, 6]], MLX::Core.squeeze(y).to_a

    z = MLX::Core.expand_dims(x, 1)
    assert_equal [2, 1, 3], z.shape
    assert_equal [[[1, 2, 3]], [[4, 5, 6]]], z.to_a
  end

  def test_atleast_nd_from_scalar
    scalar = MLX::Core.array(5, MLX::Core.int32)

    a1 = MLX::Core.atleast_1d(scalar)
    a2 = MLX::Core.atleast_2d(scalar)
    a3 = MLX::Core.atleast_3d(scalar)

    assert_equal [1], a1.shape
    assert_equal [1, 1], a2.shape
    assert_equal [1, 1, 1], a3.shape
    assert_equal [5], a1.to_a
    assert_equal [[5]], a2.to_a
    assert_equal [[[5]]], a3.to_a
  end
end
