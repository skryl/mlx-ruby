# frozen_string_literal: true

require_relative "test_helper"

class Phase178BasicDistributionLossesTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_l1_and_mse_values
    pred = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    tgt = MLX::Core.array([0.0, 2.0], MLX::Core.float32)

    assert_in_delta 0.5, MLX::NN.l1_loss(pred, tgt, reduction: "mean").item, 1e-6
    assert_in_delta 1.0, MLX::NN.l1_loss(pred, tgt, reduction: "sum").item, 1e-6
    assert_in_delta 0.5, MLX::NN.mse_loss(pred, tgt, reduction: "mean").item, 1e-6
  end

  def test_nll_and_kl_values
    inputs = MLX::Core.array([[-0.1, -2.3], [-2.0, -0.2]], MLX::Core.float32)
    targets_idx = MLX::Core.array([0, 1], MLX::Core.int32)

    nll = MLX::NN.nll_loss(inputs, targets_idx, reduction: "none")
    assert_nested_close [0.1, 0.2], nll.to_a, 1e-5

    pred_log = MLX::Core.array([[-1.0, -2.0], [-3.0, -4.0]], MLX::Core.float32)
    tgt_log = MLX::Core.array([[0.0, 0.0], [0.0, 0.0]], MLX::Core.float32)
    kl = MLX::NN.kl_div_loss(pred_log, tgt_log, reduction: "none")
    assert_nested_close [3.0, 7.0], kl.to_a, 1e-5
  end

  private

  def assert_nested_close(expected, actual, atol = 1e-5)
    assert_equal expected.length, actual.length
    expected.zip(actual).each { |e, a| assert_in_delta e, a, atol }
  end
end
