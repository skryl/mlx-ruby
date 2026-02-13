# frozen_string_literal: true

require_relative "test_helper"

class Phase126PackageParityTest < Minitest::Test
def setup
  TestSupport.build_native_extension!
  $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
  require "mlx"
end

def teardown
  $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
end

  def test_phase_126_contract
    multi = MLX::Optimizers::MultiOptimizer.new([MLX::Optimizers::SGD.new])
    assert_kind_of MLX::Optimizers::Optimizer, multi
    assert_respond_to MLX::Optimizers, :clip_grad_norm
  end
end
