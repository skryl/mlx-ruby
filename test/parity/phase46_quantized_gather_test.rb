# frozen_string_literal: true

require_relative "test_helper"

class Phase46QuantizedGatherTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_qqmm_matches_float_matmul_approximately
    x = MLX::Core.array([[0.2] * 32], MLX::Core.float32)
    w = MLX::Core.array((0...32).map { |r| (1..32).map { |c| (r + c) / 200.0 } }, MLX::Core.float32)

    w_q, scales = MLX::Core.quantize(w, nil, nil, "nvfp4")
    out = MLX::Core.qqmm(x, w_q, scales, nil, nil, "nvfp4")
    w_hat = MLX::Core.dequantize(w_q, scales, nil, nil, nil, "nvfp4", MLX::Core.float32)
    ref = MLX::Core.matmul(x, MLX::Core.transpose(w_hat))

    assert_equal ref.shape, out.shape
    assert_nested_close ref.to_a, out.to_a, 0.5
  end

  def test_gather_qmm_matches_quantized_matmul
    x = MLX::Core.array([[0.1] * 32], MLX::Core.float32)
    w = MLX::Core.array([
      (1..32).map { |i| i / 100.0 },
      (1..32).map { |i| -i / 120.0 }
    ], MLX::Core.float32)

    w_q, scales, biases = MLX::Core.quantize(w, 32, 4, "affine")

    qmm = MLX::Core.quantized_matmul(x, w_q, scales, biases, true, 32, 4, "affine")
    gqmm = MLX::Core.gather_qmm(x, w_q, scales, biases, nil, nil, true, 32, 4, "affine", false)

    assert_equal qmm.shape, gqmm.shape
    assert_nested_close qmm.to_a, gqmm.to_a, 1e-4
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
