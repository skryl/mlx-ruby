# frozen_string_literal: true

require_relative "test_helper"

class Phase181RemainingLossesTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_hinge_huber_log_cosh_and_cosine_similarity
    inputs = MLX::Core.array([2.0, -1.0], MLX::Core.float32)
    targets = MLX::Core.array([1.0, 1.0], MLX::Core.float32)
    hinge = MLX::NN.hinge_loss(inputs, targets, reduction: "mean")
    assert_in_delta 1.0, hinge.item, 1e-6

    h_in = MLX::Core.array([0.0, 2.0], MLX::Core.float32)
    h_tg = MLX::Core.array([0.0, 0.0], MLX::Core.float32)
    huber = MLX::NN.huber_loss(h_in, h_tg, delta: 1.0, reduction: "mean")
    assert_in_delta 0.75, huber.item, 1e-6

    lc = MLX::NN.log_cosh_loss(h_in, h_tg, reduction: "mean")
    assert_in_delta 0.662501, lc.item, 1e-5

    x1 = MLX::Core.array([[1.0, 0.0]], MLX::Core.float32)
    x2 = MLX::Core.array([[1.0, 0.0]], MLX::Core.float32)
    cos = MLX::NN.cosine_similarity_loss(x1, x2, axis: 1, reduction: "mean")
    assert_in_delta 1.0, cos.item, 1e-6
  end

  def test_margin_ranking_loss_and_shape_validation
    i1 = MLX::Core.array([1.0], MLX::Core.float32)
    i2 = MLX::Core.array([0.0], MLX::Core.float32)
    t1 = MLX::Core.array([1.0], MLX::Core.float32)
    t2 = MLX::Core.array([-1.0], MLX::Core.float32)

    loss_pos = MLX::NN.margin_ranking_loss(i1, i2, t1, margin: 0.0, reduction: "mean")
    loss_neg = MLX::NN.margin_ranking_loss(i1, i2, t2, margin: 0.0, reduction: "mean")

    assert_in_delta 0.0, loss_pos.item, 1e-6
    assert_in_delta 1.0, loss_neg.item, 1e-6

    assert_raises(ArgumentError) do
      MLX::NN.margin_ranking_loss(
        MLX::Core.array([1.0, 2.0], MLX::Core.float32),
        MLX::Core.array([1.0], MLX::Core.float32),
        MLX::Core.array([1.0], MLX::Core.float32)
      )
    end
  end
end
