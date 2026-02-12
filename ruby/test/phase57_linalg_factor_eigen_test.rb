# frozen_string_literal: true

require_relative "test_helper"

class Phase57LinalgFactorEigenTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_lu_and_lu_factor
    a = MLX::Core.array([[2.0, 3.0], [1.0, 4.0]], MLX::Core.float32)

    p, l, u = MLX::Core.lu(a)
    assert_equal [2], p.shape
    assert_equal [2, 2], l.shape
    assert_equal [2, 2], u.shape

    lu, pivots = MLX::Core.lu_factor(a)
    assert_equal [2, 2], lu.shape
    assert_equal [2], pivots.shape
  end

  def test_cross_and_eigen_family
    x = MLX::Core.array([1.0, 0.0, 0.0], MLX::Core.float32)
    y = MLX::Core.array([0.0, 1.0, 0.0], MLX::Core.float32)
    z = MLX::Core.cross(x, y)
    assert_nested_close [0.0, 0.0, 1.0], z.to_a

    a = MLX::Core.array([[1.0, -2.0], [-2.0, 1.0]], MLX::Core.float32)

    ev = MLX::Core.real(MLX::Core.eigvals(a)).to_a.map { |v| scalar_real(v) }.sort
    assert_in_delta(-1.0, ev[0], 1e-4)
    assert_in_delta(3.0, ev[1], 1e-4)

    e_vals, e_vecs = MLX::Core.eig(a)
    assert_equal [2], e_vals.shape
    assert_equal [2, 2], e_vecs.shape

    evh = MLX::Core.eigvalsh(a).to_a
    assert_in_delta(-1.0, evh[0], 1e-4)
    assert_in_delta(3.0, evh[1], 1e-4)

    wh, vh = MLX::Core.eigh(a)
    assert_equal [2], wh.shape
    assert_equal [2, 2], vh.shape
  end

  private

  def scalar_real(value)
    value.is_a?(Complex) ? value.real : value
  end

  def assert_nested_close(expected, actual, atol = 1e-4)
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
