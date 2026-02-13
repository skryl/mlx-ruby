# frozen_string_literal: true

require_relative "test_helper"

class Phase135PackageParityTest < Minitest::Test
def setup
  TestSupport.build_native_extension!
  $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
  require "mlx"
end

def teardown
  $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
end

  def test_phase_135_contract
    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    y = MLX::Core.array([2.0, 1.0], MLX::Core.float32)
    vars = MLX::Core.array([1.0, 1.0], MLX::Core.float32)
    logits = MLX::Core.array([[-1.0, -2.0], [-3.0, -4.0]], MLX::Core.float32)
    targets_log = MLX::Core.array([[0.0, 0.0], [0.0, 0.0]], MLX::Core.float32)
    anchors = MLX::Core.array([[0.0, 0.0]], MLX::Core.float32)
    positives = MLX::Core.array([[1.0, 0.0]], MLX::Core.float32)
    negatives = MLX::Core.array([[0.2, 0.0]], MLX::Core.float32)
    xx = MLX::Core.array([[1.0, 0.0]], MLX::Core.float32)
    yy = MLX::Core.array([[1.0, 0.0]], MLX::Core.float32)
    rank_targets = MLX::Core.array([1.0, -1.0], MLX::Core.float32)

    assert MLX::NN.gaussian_nll_loss(x, y, vars, reduction: "none").is_a?(MLX::Core::Array)
    assert MLX::NN.kl_div_loss(logits, targets_log, reduction: "none").is_a?(MLX::Core::Array)
    assert MLX::NN.smooth_l1_loss(x, y, reduction: "none").is_a?(MLX::Core::Array)
    assert MLX::NN.triplet_loss(anchors, positives, negatives, reduction: "none").is_a?(MLX::Core::Array)
    assert MLX::NN.hinge_loss(x, y, reduction: "none").is_a?(MLX::Core::Array)
    assert MLX::NN.huber_loss(x, y, reduction: "none").is_a?(MLX::Core::Array)
    assert MLX::NN.log_cosh_loss(x, y, reduction: "none").is_a?(MLX::Core::Array)
    assert MLX::NN.cosine_similarity_loss(xx, yy, reduction: "none").is_a?(MLX::Core::Array)
    assert MLX::NN.margin_ranking_loss(x, y, rank_targets, reduction: "none").is_a?(MLX::Core::Array)
  end
end
