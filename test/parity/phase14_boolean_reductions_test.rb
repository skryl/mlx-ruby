# frozen_string_literal: true

require_relative "test_helper"

class Phase14BooleanReductionsTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_all_and_any_global
    x = MLX::Core.array([[true, false, true], [true, true, false]], MLX::Core.bool_)

    assert_equal false, MLX::Core.all(x).to_a
    assert_equal true, MLX::Core.any(x).to_a
  end

  def test_all_and_any_with_axis_and_keepdims
    x = MLX::Core.array([[true, false, true], [true, true, false]], MLX::Core.bool_)

    assert_equal [true, false, false], MLX::Core.all(x, 0).to_a
    assert_equal [true, true], MLX::Core.any(x, 1).to_a

    all_keep = MLX::Core.all(x, 1, true)
    any_keep = MLX::Core.any(x, 1, true)

    assert_equal [2, 1], all_keep.shape
    assert_equal [2, 1], any_keep.shape
    assert_equal [[false], [false]], all_keep.to_a
    assert_equal [[true], [true]], any_keep.to_a
  end
end
