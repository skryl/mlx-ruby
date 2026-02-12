# frozen_string_literal: true

require_relative "test_helper"

class Phase45QuantizationBasicsTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_to_fp8_and_from_fp8_roundtrip
    vals = MLX::Core.array([448.0, 256.0, 192.0, 128.0, 96.0, 64.0, 48.0, 32.0, 24.0, 16.0, 12.0, 8.0, 6.0, 4.0, 3.0, 2.0, 0.015625], MLX::Core.float32)

    assert MLX::Core.array_equal(MLX::Core.from_fp8(MLX::Core.to_fp8(vals), MLX::Core.float32), vals)
    neg_vals = MLX::Core.negative(vals)
    assert MLX::Core.array_equal(MLX::Core.from_fp8(MLX::Core.to_fp8(neg_vals), MLX::Core.float32), neg_vals)
  end

  def test_quantize_dequantize_and_quantized_matmul_affine
    w = MLX::Core.array([
      [0.10, -0.20, 0.30, -0.40, 0.50, -0.60, 0.70, -0.80,
       0.90, -1.00, 1.10, -1.20, 1.30, -1.40, 1.50, -1.60,
       1.70, -1.80, 1.90, -2.00, 2.10, -2.20, 2.30, -2.40,
       2.50, -2.60, 2.70, -2.80, 2.90, -3.00, 3.10, -3.20],
      [-0.15, 0.25, -0.35, 0.45, -0.55, 0.65, -0.75, 0.85,
       -0.95, 1.05, -1.15, 1.25, -1.35, 1.45, -1.55, 1.65,
       -1.75, 1.85, -1.95, 2.05, -2.15, 2.25, -2.35, 2.45,
       -2.55, 2.65, -2.75, 2.85, -2.95, 3.05, -3.15, 3.25]
    ], MLX::Core.float32)

    w_q, scales, biases = MLX::Core.quantize(w, 32, 4, "affine")
    assert_equal [2, 4], w_q.shape

    w_hat = MLX::Core.dequantize(w_q, scales, biases, 32, 4, "affine", MLX::Core.float32)
    assert_equal w.shape, w_hat.shape

    x = MLX::Core.array([[0.5] * 32], MLX::Core.float32)
    out_q = MLX::Core.quantized_matmul(x, w_q, scales, biases, true, 32, 4, "affine")
    out_ref = MLX::Core.matmul(x, MLX::Core.transpose(w_hat))

    assert_equal out_ref.shape, out_q.shape
    assert_nested_close out_ref.to_a, out_q.to_a, 0.35
  end

  private

  def assert_nested_close(expected, actual, atol = 1e-4)
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
