# frozen_string_literal: true

require_relative "test_helper"

class Phase17MeshgridTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_meshgrid_default_xy
    x = MLX::Core.array([1, 2, 3], MLX::Core.int32)
    y = MLX::Core.array([10, 20], MLX::Core.int32)

    grids = MLX::Core.meshgrid([x, y])

    assert_equal 2, grids.length
    assert_equal [2, 3], grids[0].shape
    assert_equal [2, 3], grids[1].shape
    assert_equal [[1, 2, 3], [1, 2, 3]], grids[0].to_a
    assert_equal [[10, 10, 10], [20, 20, 20]], grids[1].to_a
  end

  def test_meshgrid_ij_and_sparse
    x = MLX::Core.array([1, 2, 3], MLX::Core.int32)
    y = MLX::Core.array([10, 20], MLX::Core.int32)

    ij = MLX::Core.meshgrid([x, y], false, "ij")
    assert_equal [3, 2], ij[0].shape
    assert_equal [3, 2], ij[1].shape
    assert_equal [[1, 1], [2, 2], [3, 3]], ij[0].to_a
    assert_equal [[10, 20], [10, 20], [10, 20]], ij[1].to_a

    sparse = MLX::Core.meshgrid([x, y], true)
    assert_equal [1, 3], sparse[0].shape
    assert_equal [2, 1], sparse[1].shape
  end
end
