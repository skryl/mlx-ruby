# frozen_string_literal: true

require_relative "test_helper"

class Phase32DiagTraceTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_diag_for_vector_and_matrix
    v = MLX::Core.array([1, 2, 3], MLX::Core.int32)
    m = MLX::Core.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], MLX::Core.int32)

    assert_equal [[1, 0, 0], [0, 2, 0], [0, 0, 3]], MLX::Core.diag(v).to_a
    assert_equal [1, 5, 9], MLX::Core.diag(m).to_a
  end

  def test_diagonal_and_trace
    m = MLX::Core.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], MLX::Core.int32)

    assert_equal [1, 5, 9], MLX::Core.diagonal(m).to_a
    assert_equal [2, 6], MLX::Core.diagonal(m, 1).to_a
    assert_equal 15, MLX::Core.trace(m).to_a
    assert_equal 8, MLX::Core.trace(m, 1).to_a
  end
end
