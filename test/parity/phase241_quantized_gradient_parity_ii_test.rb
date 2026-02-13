# frozen_string_literal: true

require_relative "test_helper"

class Phase241QuantizedGradientParityIITest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
    MLX::Core.random_seed(17)
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_gather_qmm_grad_matches_shape_expectation
    x = MLX::Core.random_uniform([2, 64], -1.0, 1.0, MLX::Core.float32)
    w = MLX::Core.random_uniform([32, 64], -1.0, 1.0, MLX::Core.float32)
    wq, scales, biases = MLX::Core.quantize(w, 32, 4, "affine")

    fn = ->(v) { MLX::Core.gather_qmm(v, wq, scales, biases, nil, nil, true, 32, 4, "affine", false) }
    out = fn.call(x)
    _, vjps = MLX::Core.vjp(fn, [x], [MLX::Core.ones_like(out)])
    assert_equal x.shape, vjps[0].shape
  end

  def test_affine_vjp_wrt_scales_and_biases
    x = MLX::Core.random_uniform([2, 64], -1.0, 1.0, MLX::Core.float32)
    w = MLX::Core.random_uniform([32, 64], -1.0, 1.0, MLX::Core.float32)
    wq, scales, biases = MLX::Core.quantize(w, 32, 4, "affine")

    grads = MLX::Core.grad(lambda do |s, b|
      y = MLX::Core.quantized_matmul(x, wq, s, b, true, 32, 4, "affine")
      MLX::Core.sum(y)
    end, [0, 1]).call(scales, biases)

    assert_equal scales.shape, grads[0].shape
    assert_equal biases.shape, grads[1].shape
  end

  def test_fp_vjp_scales_throws_for_mxfp8
    x = MLX::Core.random_uniform([1, 64], -1.0, 1.0, MLX::Core.float32)
    w = MLX::Core.random_uniform([32, 64], -1.0, 1.0, MLX::Core.float32)
    wq, scales = MLX::Core.quantize(w, nil, nil, "mxfp8")

    err = assert_raises(RuntimeError) do
      MLX::Core.grad(lambda do |s|
        y = MLX::Core.quantized_matmul(x, wq, s, nil, true, nil, nil, "mxfp8")
        MLX::Core.sum(y)
      end).call(scales)
    end
    assert_match(/no gradient wrt scales/i, err.message)
  end

  def test_quantize_strided_array
    w = MLX::Core.random_uniform([64, 64], -1.0, 1.0, MLX::Core.float32)
    strided = MLX::Core.swapaxes(w, 0, 1)
    wq, scales, biases = MLX::Core.quantize(strided, 32, 4, "affine")
    w_hat = MLX::Core.dequantize(wq, scales, biases, 32, 4, "affine", MLX::Core.float32)

    assert_equal [64, 64], w_hat.shape
  end
end
