# frozen_string_literal: true

require_relative "test_helper"

class Phase115PackageParityTest < Minitest::Test
def setup
  TestSupport.build_native_extension!
  $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
  require "mlx"
end

def teardown
  $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
end

  def test_phase_115_contract
    opt = MLX::Optimizers::Optimizer.new(learning_rate: 0.1)
    assert_respond_to opt, :update
    assert_equal 0, opt.step
  end
end
