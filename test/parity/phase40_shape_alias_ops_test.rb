# frozen_string_literal: true

require_relative "test_helper"

class Phase40ShapeAliasOpsTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_asarray_and_concat_alias
    a = MLX::Core.asarray([1, 2, 3], MLX::Core.int32)
    assert_equal [1, 2, 3], a.to_a
    assert_equal :int32, a.dtype.name

    left = MLX::Core.array([[1, 2]], MLX::Core.int32)
    right = MLX::Core.array([[3, 4]], MLX::Core.int32)
    assert_equal [[1, 2], [3, 4]], MLX::Core.concat([left, right], 0).to_a
  end

  def test_permute_unflatten_view_and_as_strided
    x = MLX::Core.reshape(MLX::Core.arange(6, MLX::Core.int32), [2, 3])

    p = MLX::Core.permute_dims(x, [1, 0])
    assert_equal [3, 2], p.shape
    assert_equal [[0, 3], [1, 4], [2, 5]], p.to_a

    flat = MLX::Core.flatten(x)
    restored = MLX::Core.unflatten(flat, 0, [2, 3])
    assert_equal [2, 3], restored.shape
    assert_equal [[0, 1, 2], [3, 4, 5]], restored.to_a

    viewed = MLX::Core.view(x, MLX::Core.uint32)
    assert_equal :uint32, viewed.dtype.name
    assert_equal [[0, 1, 2], [3, 4, 5]], viewed.to_a

    stride_src = MLX::Core.array([[1, 2, 3], [4, 5, 6]], MLX::Core.int32)
    window = MLX::Core.as_strided(stride_src, [2, 2], [3, 1], 1)
    assert_equal [[2, 3], [5, 6]], window.to_a
  end

  def test_broadcast_shapes
    assert_equal [3, 1], MLX::Core.broadcast_shapes([1], [3, 1])
    assert_equal [5, 3, 4], MLX::Core.broadcast_shapes([5, 1, 4], [1, 3, 1])
  end
end
