# frozen_string_literal: true

require_relative "test_helper"

class Phase130PackageParityTest < Minitest::Test
def setup
  TestSupport.build_native_extension!
  $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
  require "mlx"
end

def teardown
  $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
end

  def test_phase_130_contract
    mod = MLX::NN::Module.new
    assert_equal false, mod.eval.training
    assert_equal true, mod.train.training
  end
end
