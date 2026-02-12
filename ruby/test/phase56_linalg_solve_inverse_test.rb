# frozen_string_literal: true

require_relative "test_helper"

class Phase56LinalgSolveInverseTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_inv_solve_and_solve_triangular
    a = MLX::Core.array([[4.0, 7.0], [2.0, 6.0]], MLX::Core.float32)
    ainv = MLX::Core.inv(a)
    eye = MLX::Core.matmul(a, ainv)
    assert_nested_close [[1.0, 0.0], [0.0, 1.0]], eye.to_a

    b = MLX::Core.array([1.0, 0.0], MLX::Core.float32)
    x = MLX::Core.solve(a, b)
    ax = MLX::Core.matmul(a, MLX::Core.reshape(x, [2, 1]))
    assert_nested_close [[1.0], [0.0]], ax.to_a

    lower = MLX::Core.array([[2.0, 0.0], [3.0, 1.0]], MLX::Core.float32)
    rhs = MLX::Core.array([4.0, 7.0], MLX::Core.float32)
    sol = MLX::Core.solve_triangular(lower, rhs)
    check = MLX::Core.matmul(lower, MLX::Core.reshape(sol, [2, 1]))
    assert_nested_close [[4.0], [7.0]], check.to_a

    tinv = MLX::Core.tri_inv(lower)
    eye_tri = MLX::Core.matmul(lower, tinv)
    assert_nested_close [[1.0, 0.0], [0.0, 1.0]], eye_tri.to_a
  end

  def test_cholesky_cholesky_inv_and_pinv
    a = MLX::Core.array([[4.0, 2.0], [2.0, 3.0]], MLX::Core.float32)

    l = MLX::Core.cholesky(a)
    lt = MLX::Core.transpose(l)
    reconstructed = MLX::Core.matmul(l, lt)
    assert_nested_close a.to_a, reconstructed.to_a

    a_inv_from_chol = MLX::Core.cholesky_inv(l)
    eye = MLX::Core.matmul(a, a_inv_from_chol)
    assert_nested_close [[1.0, 0.0], [0.0, 1.0]], eye.to_a

    rect = MLX::Core.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], MLX::Core.float32)
    p = MLX::Core.pinv(rect)
    assert_equal [3, 2], p.shape

    approx = MLX::Core.matmul(MLX::Core.matmul(rect, p), rect)
    assert_nested_close rect.to_a, approx.to_a
  end

  private

  def assert_nested_close(expected, actual, atol = 2e-3)
    assert_equal structure_signature(expected), structure_signature(actual)
    flatten(expected).zip(flatten(actual)).each do |exp, got|
      assert_in_delta exp, got, atol
    end
  end

  def flatten(value)
    return [value] unless value.is_a?(Array)

    value.flat_map { |v| flatten(v) }
  end

  def structure_signature(value)
    return :scalar unless value.is_a?(Array)

    [value.length, *(value.map { |v| structure_signature(v) })]
  end
end
