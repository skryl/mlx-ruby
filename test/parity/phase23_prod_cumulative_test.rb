# frozen_string_literal: true

require_relative "test_helper"

class Phase23ProdCumulativeTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_prod_reduction
    x = MLX::Core.array([[1, 2, 3], [4, 5, 6]], MLX::Core.int32)

    assert_equal 720, MLX::Core.prod(x).to_a
    assert_equal [4, 10, 18], MLX::Core.prod(x, 0).to_a

    kept = MLX::Core.prod(x, 1, true)
    assert_equal [2, 1], kept.shape
    assert_equal [[6], [120]], kept.to_a
  end

  def test_cumsum_and_cumprod
    x = MLX::Core.array([[1, 2, 3], [4, 5, 6]], MLX::Core.int32)

    assert_equal [1, 3, 6, 10, 15, 21], MLX::Core.cumsum(x).to_a
    assert_equal [[1, 3, 6], [4, 9, 15]], MLX::Core.cumsum(x, 1).to_a

    assert_equal [1, 2, 6, 24, 120, 720], MLX::Core.cumprod(x).to_a
    assert_equal [[1, 2, 6], [4, 20, 120]], MLX::Core.cumprod(x, 1).to_a
  end
end
