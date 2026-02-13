# frozen_string_literal: true

require_relative "test_helper"

class Phase233CompileShapelessOpsParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_shapeless_compile_unflatten_gather_full_like
    gather_idx = MLX::Core.array([[1, 0], [0, 1]], MLX::Core.int32)
    compiled = MLX::Core.compile(lambda do |x|
      flat = MLX::Core.reshape(x, [x.shape.reduce(:*)])
      rebuilt = MLX::Core.reshape(flat, x.shape)
      gathered = MLX::Core.take_along_axis(rebuilt, gather_idx, 1)
      [rebuilt, gathered, MLX::Core.full_like(rebuilt, 2.0)]
    end, nil, nil, true)

    x = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)
    rebuilt, gathered, filled = compiled.call(x)
    assert_equal [[1.0, 2.0], [3.0, 4.0]], rebuilt.to_a
    assert_equal [[2.0, 1.0], [3.0, 4.0]], gathered.to_a
    assert_equal [[2.0, 2.0], [2.0, 2.0]], filled.to_a
  end

  def test_shapeless_compile_matmul_slice_update_and_reshape
    w = MLX::Core.array([[1.0], [2.0]], MLX::Core.float32)
    update = MLX::Core.array([[9.0, 9.0]], MLX::Core.float32)
    compiled = MLX::Core.compile(lambda do |x|
      mm = MLX::Core.matmul(x, w)
      patched = MLX::Core.slice_update(x, update, [0, 0], [1, 2])
      [mm, MLX::Core.reshape(patched, [4])]
    end, nil, nil, true)

    x = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)
    mm, flat = compiled.call(x)
    assert_equal [[5.0], [11.0]], mm.to_a
    assert_equal [9.0, 9.0, 3.0, 4.0], flat.to_a
  end
end
