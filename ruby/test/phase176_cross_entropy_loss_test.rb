# frozen_string_literal: true

require_relative "test_helper"

class Phase176CrossEntropyLossTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_cross_entropy_with_class_indices_and_probs
    logits = MLX::Core.array([[2.0, -1.0], [-1.0, 2.0]], MLX::Core.float32)
    targets_idx = MLX::Core.array([0, 1], MLX::Core.int32)
    targets_prob = MLX::Core.array([[0.9, 0.1], [0.1, 0.9]], MLX::Core.float32)

    loss_idx = MLX::NN.cross_entropy(logits, targets_idx, reduction: "none")
    loss_prob = MLX::NN.cross_entropy(logits, targets_prob, reduction: "none")

    assert_nested_close [0.0485873, 0.0485873], loss_idx.to_a, 1e-5
    assert_nested_close [0.348587, 0.348587], loss_prob.to_a, 1e-5

    mean_loss = MLX::NN.cross_entropy(logits, targets_idx, reduction: "mean")
    assert_in_delta 0.0485873, mean_loss.item, 1e-5
  end

  def test_cross_entropy_validates_shapes_weights_and_label_smoothing
    logits = MLX::Core.array([[2.0, -1.0], [-1.0, 2.0]], MLX::Core.float32)
    targets = MLX::Core.array([0, 1], MLX::Core.int32)

    assert_raises(ArgumentError) do
      MLX::NN.cross_entropy(logits, targets, label_smoothing: -0.1)
    end

    assert_raises(ArgumentError) do
      bad_targets = MLX::Core.array([[0, 1]], MLX::Core.int32)
      MLX::NN.cross_entropy(logits, bad_targets)
    end

    assert_raises(ArgumentError) do
      bad_weights = MLX::Core.array([1.0], MLX::Core.float32)
      MLX::NN.cross_entropy(logits, targets, weights: bad_weights)
    end
  end

  private

  def assert_nested_close(expected, actual, atol = 1e-5)
    assert_equal expected.length, actual.length
    expected.zip(actual).each { |e, a| assert_in_delta e, a, atol }
  end
end
