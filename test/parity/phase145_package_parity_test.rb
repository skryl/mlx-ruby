# frozen_string_literal: true

require_relative "test_helper"

class Phase145PackageParityTest < Minitest::Test
def setup
  TestSupport.build_native_extension!
  $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
  require "mlx"
end

def teardown
  $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
end

  def test_phase_145_contract
    assert MLX::NN.const_defined?(:Module)
    assert MLX::NN.const_defined?(:Linear)
    assert_respond_to MLX::NN, :value_and_grad
  end
end
