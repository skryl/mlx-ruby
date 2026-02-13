# frozen_string_literal: true

require_relative "test_helper"

class Phase21SortArgsortTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_sort_default_and_axis
    x = MLX::Core.array([[3, 1, 2], [0, -1, 4]], MLX::Core.int32)

    assert_equal [-1, 0, 1, 2, 3, 4], MLX::Core.sort(x).to_a
    assert_equal [[1, 2, 3], [-1, 0, 4]], MLX::Core.sort(x, 1).to_a
  end

  def test_argsort_default_and_axis
    x = MLX::Core.array([[3, 1, 2], [0, -1, 4]], MLX::Core.int32)

    assert_equal [4, 3, 1, 2, 0, 5], MLX::Core.argsort(x).to_a
    assert_equal [[1, 2, 0], [1, 0, 2]], MLX::Core.argsort(x, 1).to_a
  end
end
