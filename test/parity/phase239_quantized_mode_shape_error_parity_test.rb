# frozen_string_literal: true

require_relative "test_helper"

class Phase239QuantizedModeShapeErrorParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
    MLX::Core.random_seed(11)
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_mode_error_cases
    w = MLX::Core.random_uniform([256, 256], -1.0, 1.0, MLX::Core.float32)
    x = MLX::Core.random_uniform([1, 256], -1.0, 1.0, MLX::Core.float32)

    assert_raises(RuntimeError) { MLX::Core.quantize(w, nil, nil, "xyz") }

    wq, scales, biases = MLX::Core.quantize(w, 32, 4, "affine")
    assert_raises(RuntimeError) { MLX::Core.dequantize(wq, scales, biases, 32, 4, "xyz", MLX::Core.float32) }
    assert_raises(RuntimeError) { MLX::Core.quantized_matmul(x, wq, scales, biases, true, 32, 4, "xyz") }
  end

  def test_non_multiple_dimensions_for_nvfp4
    x = MLX::Core.random_uniform([1, 48], -1.0, 1.0, MLX::Core.float32)
    w = MLX::Core.random_uniform([128, 48], -1.0, 1.0, MLX::Core.float32)

    wq, scales = MLX::Core.quantize(w, nil, nil, "nvfp4")
    y_q = MLX::Core.quantized_matmul(x, wq, scales, nil, true, nil, nil, "nvfp4")
    w_hat = MLX::Core.dequantize(wq, scales, nil, nil, nil, "nvfp4", MLX::Core.float32)
    y_hat = MLX::Core.matmul(x, MLX::Core.swapaxes(w_hat, -1, -2))

    assert_equal y_hat.shape, y_q.shape
    assert_operator max_abs_diff(y_q, y_hat), :<, 2e-3
  end

  private

  def max_abs_diff(lhs, rhs)
    MLX::Core.max(MLX::Core.abs(MLX::Core.subtract(lhs, rhs))).to_a
  end
end
