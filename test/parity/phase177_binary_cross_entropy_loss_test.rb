# frozen_string_literal: true

require_relative "test_helper"

class Phase177BinaryCrossEntropyLossTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
  end

  def teardown
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_binary_cross_entropy_logits_and_probabilities
    logits = MLX::Core.array([0.105361, 0.223144, 1.20397, 0.916291], MLX::Core.float32)
    probs = MLX::Core.array([0.1, 0.1, 0.4, 0.4], MLX::Core.float32)
    targets = MLX::Core.array([0.0, 0.0, 1.0, 1.0], MLX::Core.float32)

    loss_logits = MLX::NN.binary_cross_entropy(logits, targets, with_logits: true, reduction: "mean")
    loss_probs = MLX::NN.binary_cross_entropy(probs, targets, with_logits: false, reduction: "mean")

    assert_in_delta 0.539245, loss_logits.item, 1e-5
    assert_in_delta 0.510826, loss_probs.item, 1e-5
  end

  def test_binary_cross_entropy_shape_and_weight_validation
    inputs = MLX::Core.array([0.1, 0.2], MLX::Core.float32)
    targets = MLX::Core.array([0.0], MLX::Core.float32)

    assert_raises(ArgumentError) do
      MLX::NN.binary_cross_entropy(inputs, targets)
    end

    inputs2 = MLX::Core.array([0.1, 0.2], MLX::Core.float32)
    targets2 = MLX::Core.array([0.0, 1.0], MLX::Core.float32)
    bad_weights = MLX::Core.array([1.0], MLX::Core.float32)

    assert_raises(ArgumentError) do
      MLX::NN.binary_cross_entropy(inputs2, targets2, weights: bad_weights)
    end
  end
end
