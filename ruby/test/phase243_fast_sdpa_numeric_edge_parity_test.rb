# frozen_string_literal: true

require_relative "test_helper"

class Phase243FastSdpaNumericEdgeParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
    @previous_device = MLX::Core.default_device
    MLX::Core.set_default_device(MLX::Core.cpu)
  end

  def teardown
    MLX::Core.set_default_device(@previous_device) if defined?(@previous_device) && @previous_device
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_sdpa_fully_masked_and_inf_score_paths_stay_finite
    q = MLX::Core.array([[[[1.0e4, -1.0e4], [1.0e4, -1.0e4]]]], MLX::Core.float32)
    k = MLX::Core.array([[[[1.0e4, -1.0e4], [-1.0e4, 1.0e4]]]], MLX::Core.float32)
    v = MLX::Core.array([[[[1.0, 2.0], [3.0, 4.0]]]], MLX::Core.float32)

    masked = MLX::Core.full([1, 1, 2, 2], -1.0e9, MLX::Core.float32)
    out_masked = MLX::Core.scaled_dot_product_attention(q, k, v, 1.0, masked)
    assert_equal false, MLX::Core.any(MLX::Core.isnan(out_masked)).to_a

    out_inf = MLX::Core.scaled_dot_product_attention(q, k, v, 1.0, nil)
    assert_equal false, MLX::Core.any(MLX::Core.isnan(out_inf)).to_a
  end

  def test_sdpa_few_query_and_sliced_inputs
    q = MLX::Core.array([[[[0.1, 0.2]]]], MLX::Core.float32) # T_q = 1
    k = MLX::Core.array([[[[0.2, 0.1], [0.4, 0.3], [0.5, 0.7]]]], MLX::Core.float32)
    v = MLX::Core.array([[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]], MLX::Core.float32)
    out = MLX::Core.scaled_dot_product_attention(q, k, v, 1.0, nil)
    assert_equal [1, 1, 1, 2], out.shape

    big_q = MLX::Core.array([[[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]]], MLX::Core.float32)
    sliced_q = big_q.__getitem__(0)
    sliced_q = MLX::Core.reshape(sliced_q, [1, 1, 3, 2])
    out_sliced = MLX::Core.scaled_dot_product_attention(sliced_q, k, v, 1.0, nil)
    assert_equal [1, 1, 3, 2], out_sliced.shape
  end
end
