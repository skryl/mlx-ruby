# frozen_string_literal: true

require_relative "test_helper"

class Phase180SmoothL1TripletLossTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_smooth_l1_values
    pred = MLX::Core.array([0.0, 2.0], MLX::Core.float32)
    tgt = MLX::Core.array([0.0, 0.0], MLX::Core.float32)

    loss = MLX::NN.smooth_l1_loss(pred, tgt, beta: 1.0, reduction: "mean")
    assert_in_delta 0.75, loss.item, 1e-6
  end

  def test_triplet_loss_values
    anchors = MLX::Core.array([[0.0, 0.0]], MLX::Core.float32)
    positives = MLX::Core.array([[1.0, 0.0]], MLX::Core.float32)
    negatives = MLX::Core.array([[0.2, 0.0]], MLX::Core.float32)

    loss = MLX::NN.triplet_loss(
      anchors,
      positives,
      negatives,
      axis: -1,
      p: 2,
      margin: 1.0,
      eps: 1e-6,
      reduction: "mean"
    )

    assert_in_delta 1.8, loss.item, 1e-3
  end
end
