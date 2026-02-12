# frozen_string_literal: true

require_relative "test_helper"

class Phase240QuantizedGradientParityITest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
    MLX::Core.random_seed(13)
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_qmm_vjp_matches_quantized_matmul_of_cotangent
    x = MLX::Core.random_uniform([2, 64, 64], -1.0, 1.0, MLX::Core.float32)
    w = MLX::Core.random_uniform([128, 64], -1.0, 1.0, MLX::Core.float32)
    wq, scales, biases = MLX::Core.quantize(w, 64, 8, "affine")

    [true, false].each do |transposed|
      weight = transposed ? w : MLX::Core.swapaxes(w, -1, -2)
      wq_t, scales_t, biases_t = MLX::Core.quantize(weight, 64, 8, "affine")
      fn = ->(v) { MLX::Core.quantized_matmul(v, wq_t, scales_t, biases_t, transposed, 64, 8, "affine") }
      out = fn.call(x)
      cot = MLX::Core.ones_like(out)

      _, vjps = MLX::Core.vjp(fn, [x], [cot])
      expected = MLX::Core.quantized_matmul(cot, wq_t, scales_t, biases_t, !transposed, 64, 8, "affine")
      assert_operator max_abs_diff(vjps[0], expected), :<, 1e-3
    end
  end

  def test_qmm_jvp_matches_quantized_matmul_of_tangent
    x = MLX::Core.random_uniform([2, 64, 64], -1.0, 1.0, MLX::Core.float32)
    tangent = MLX::Core.ones_like(x)
    w = MLX::Core.random_uniform([128, 64], -1.0, 1.0, MLX::Core.float32)
    wq, scales, biases = MLX::Core.quantize(w, 64, 8, "affine")

    fn = ->(v) { MLX::Core.quantized_matmul(v, wq, scales, biases, true, 64, 8, "affine") }
    _, jvps = MLX::Core.jvp(fn, [x], [tangent])
    expected = MLX::Core.quantized_matmul(tangent, wq, scales, biases, true, 64, 8, "affine")
    assert_operator max_abs_diff(jvps[0], expected), :<, 1e-3
  end

  private

  def max_abs_diff(lhs, rhs)
    MLX::Core.max(MLX::Core.abs(MLX::Core.subtract(lhs, rhs))).to_a
  end
end
