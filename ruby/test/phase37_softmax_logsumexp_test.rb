# frozen_string_literal: true

require_relative "test_helper"

class Phase37SoftmaxLogsumexpTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_sigmoid_and_softmax
    x = MLX::Core.array([-MLX::Core.inf, 0.0, MLX::Core.inf], MLX::Core.float32)
    assert_nested_close [0.0, 0.5, 1.0], MLX::Core.sigmoid(x).to_a, 1e-4

    v = MLX::Core.array([1.0, 2.0, 3.0], MLX::Core.float32)
    p = MLX::Core.softmax(v).to_a
    assert_in_delta 1.0, p.sum, 1e-5
    assert p[2] > p[1]
    assert p[1] > p[0]

    m = MLX::Core.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], MLX::Core.float32)
    row_softmax = MLX::Core.softmax(m, 1).to_a
    row_softmax.each do |row|
      assert_in_delta 1.0, row.sum, 1e-5
    end
  end

  def test_logsumexp_and_logcumsumexp
    v = MLX::Core.array([0.0, 0.0], MLX::Core.float32)
    assert_in_delta Math.log(2.0), MLX::Core.logsumexp(v).to_a, 1e-5

    m = MLX::Core.array([[0.0, 0.0], [0.0, 0.0]], MLX::Core.float32)
    assert_nested_close [Math.log(2.0), Math.log(2.0)], MLX::Core.logsumexp(m, 1).to_a, 1e-5

    c = MLX::Core.array([0.0, 0.0, 0.0], MLX::Core.float32)
    assert_nested_close [0.0, Math.log(2.0), Math.log(3.0)], MLX::Core.logcumsumexp(c).to_a, 1e-5
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
