# frozen_string_literal: true

require_relative "test_helper"

class Phase41SpecialMathCumextTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_divmod_logaddexp_and_rsqrt
    a = MLX::Core.array([7.0, 8.0, 9.0], MLX::Core.float32)
    q, r = MLX::Core.divmod(a, 2.0)
    assert_nested_close [3.0, 4.0, 4.0], q.to_a
    assert_nested_close [1.0, 0.0, 1.0], r.to_a

    x = MLX::Core.array([0.0, 1.0], MLX::Core.float32)
    y = MLX::Core.array([0.0, 2.0], MLX::Core.float32)
    expected = [Math.log(Math.exp(0.0) + Math.exp(0.0)), Math.log(Math.exp(1.0) + Math.exp(2.0))]
    assert_nested_close expected, MLX::Core.logaddexp(x, y).to_a

    assert_nested_close [1.0, 0.5, 1.0 / 3.0], MLX::Core.rsqrt(MLX::Core.array([1.0, 4.0, 9.0], MLX::Core.float32)).to_a
  end

  def test_erf_erfinv_and_inverse_hyperbolic
    vals = MLX::Core.array([-1.0, -0.5, 0.0, 0.5, 1.0], MLX::Core.float32)
    erf_vals = MLX::Core.erf(vals)
    roundtrip = MLX::Core.erfinv(erf_vals)
    assert_nested_close vals.to_a, roundtrip.to_a, 1e-3

    x = MLX::Core.array([-2.0, -1.0, 0.0, 1.0, 2.0], MLX::Core.float32)
    assert_nested_close x.to_a, MLX::Core.sinh(MLX::Core.arcsinh(x)).to_a, 1e-4

    y = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)
    assert_nested_close y.to_a, MLX::Core.cosh(MLX::Core.arccosh(y)).to_a, 1e-4

    z = MLX::Core.array([-0.8, 0.0, 0.8], MLX::Core.float32)
    assert_nested_close z.to_a, MLX::Core.tanh(MLX::Core.arctanh(z)).to_a, 1e-4
  end

  def test_cummax_and_cummin
    v = MLX::Core.array([3.0, 1.0, 4.0, 2.0], MLX::Core.float32)
    assert_nested_close [3.0, 3.0, 4.0, 4.0], MLX::Core.cummax(v).to_a
    assert_nested_close [3.0, 1.0, 1.0, 1.0], MLX::Core.cummin(v).to_a

    m = MLX::Core.array([[3.0, 1.0, 4.0], [2.0, 5.0, 0.0]], MLX::Core.float32)
    assert_nested_close [[3.0, 3.0, 4.0], [2.0, 5.0, 5.0]], MLX::Core.cummax(m, 1).to_a
    assert_nested_close [[3.0, 1.0, 1.0], [2.0, 2.0, 0.0]], MLX::Core.cummin(m, 1).to_a
  end

  private

  def assert_nested_close(expected, actual, atol = 1e-5)
    assert_equal structure_signature(expected), structure_signature(actual)
    flatten(expected).zip(flatten(actual)).each do |exp, got|
      assert_in_delta exp, got, atol
    end
  end

  def flatten(value)
    return [value] unless value.is_a?(Array)

    value.flat_map { |x| flatten(x) }
  end

  def structure_signature(value)
    return :scalar unless value.is_a?(Array)

    [value.length, *(value.map { |v| structure_signature(v) })]
  end
end
