# frozen_string_literal: true

require_relative "test_helper"

class Phase42IndexUpdatePadTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_pad_constant
    x = MLX::Core.array([[1, 2], [3, 4]], MLX::Core.int32)

    out = MLX::Core.pad(x, 1)
    assert_equal [[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]], out.to_a

    v = MLX::Core.array([1, 2], MLX::Core.int32)
    out2 = MLX::Core.pad(v, [1, 2], "constant", 9)
    assert_equal [9, 1, 2, 9, 9], out2.to_a
  end

  def test_slice_update
    src = MLX::Core.zeros([3, 3], MLX::Core.int32)
    upd = MLX::Core.array([[1, 2], [3, 4]], MLX::Core.int32)

    out = MLX::Core.slice_update(src, upd, [1, 1], [3, 3])
    assert_equal [[0, 0, 0], [0, 1, 2], [0, 3, 4]], out.to_a
  end

  def test_put_along_axis
    a = MLX::Core.array([[10, 20, 30], [40, 50, 60]], MLX::Core.int32)
    idx = MLX::Core.array([[2, 0], [1, 2]], MLX::Core.int32)
    vals = MLX::Core.array([[7, 8], [9, 10]], MLX::Core.int32)

    out = MLX::Core.put_along_axis(a, idx, vals, 1)
    assert_equal [[8, 20, 7], [40, 9, 10]], out.to_a
  end
end
