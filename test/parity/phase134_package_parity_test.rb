# frozen_string_literal: true

require_relative "test_helper"

class Phase134PackageParityTest < Minitest::Test
def setup
  TestSupport.build_native_extension!
  $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
  require "mlx"
end

def teardown
  $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
end

  def test_phase_134_contract
    logits = MLX::Core.array([[2.0, -1.0], [-1.0, 2.0]], MLX::Core.float32)
    idx = MLX::Core.array([0, 1], MLX::Core.int32)
    probs = MLX::Core.array([0.1, 0.9], MLX::Core.float32)
    targets = MLX::Core.array([0.0, 1.0], MLX::Core.float32)
    x = MLX::Core.array([1.0, 2.0], MLX::Core.float32)
    y = MLX::Core.array([2.0, 1.0], MLX::Core.float32)

    assert MLX::NN.cross_entropy(logits, idx, reduction: "none").is_a?(MLX::Core::Array)
    assert MLX::NN.binary_cross_entropy(probs, targets, with_logits: false, reduction: "none").is_a?(MLX::Core::Array)
    assert MLX::NN.l1_loss(x, y, reduction: "none").is_a?(MLX::Core::Array)
    assert MLX::NN.mse_loss(x, y, reduction: "none").is_a?(MLX::Core::Array)
    assert MLX::NN.nll_loss(logits, idx, reduction: "none").is_a?(MLX::Core::Array)
  end
end
