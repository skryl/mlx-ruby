# frozen_string_literal: true

require_relative "test_helper"

class Phase238QuantizedMatvecScenariosParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
    MLX::Core.random_seed(7)
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_quantized_matmul_qmv_like_path
    x = MLX::Core.random_uniform([3, 1, 64], -1.0, 1.0, MLX::Core.float32)
    w = MLX::Core.random_uniform([3, 67, 64], -1.0, 1.0, MLX::Core.float32)
    wq, scales, biases = MLX::Core.quantize(w, 32, 4, "affine")
    w_hat = MLX::Core.dequantize(wq, scales, biases, 32, 4, "affine", MLX::Core.float32)

    y_q = MLX::Core.quantized_matmul(x, wq, scales, biases, true, 32, 4, "affine")
    y_hat = MLX::Core.matmul(x, MLX::Core.swapaxes(w_hat, -1, -2))
    assert_equal y_hat.shape, y_q.shape
    assert_operator max_abs_diff(y_q, y_hat), :<, 1e-3
  end

  def test_quantized_matmul_qvm_like_path
    x = MLX::Core.random_uniform([2, 1, 64], -1.0, 1.0, MLX::Core.float32)
    w = MLX::Core.random_uniform([2, 64, 32], -1.0, 1.0, MLX::Core.float32)
    wq, scales, biases = MLX::Core.quantize(w, 32, 4, "affine")
    w_hat = MLX::Core.dequantize(wq, scales, biases, 32, 4, "affine", MLX::Core.float32)

    y_q = MLX::Core.quantized_matmul(x, wq, scales, biases, false, 32, 4, "affine")
    y_hat = MLX::Core.matmul(x, w_hat)
    assert_equal y_hat.shape, y_q.shape
    assert_operator max_abs_diff(y_q, y_hat), :<, 1e-3
  end

  def test_qqmm_mode_path_matches_dequantized_reference
    x = MLX::Core.random_uniform([1, 64], -1.0, 1.0, MLX::Core.float32)
    w = MLX::Core.random_uniform([32, 64], -1.0, 1.0, MLX::Core.float32)
    wq, scales = MLX::Core.quantize(w, nil, nil, "mxfp8")
    w_hat = MLX::Core.dequantize(wq, scales, nil, nil, nil, "mxfp8", MLX::Core.float32)

    y_q = MLX::Core.qqmm(x, wq, scales, nil, nil, "mxfp8")
    y_hat = MLX::Core.matmul(x, MLX::Core.swapaxes(w_hat, -1, -2))
    assert_equal y_hat.shape, y_q.shape
    assert_operator max_abs_diff(y_q, y_hat), :<, 5e-1
  end

  private

  def max_abs_diff(lhs, rhs)
    MLX::Core.max(MLX::Core.abs(MLX::Core.subtract(lhs, rhs))).to_a
  end
end
