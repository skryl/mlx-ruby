# frozen_string_literal: true

require_relative "test_helper"

class Phase127PackageParityTest < Minitest::Test
def setup
  TestSupport.build_native_extension!
  $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
  require "mlx"
end

def teardown
  $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
end

  def test_phase_127_contract
    mod = MLX::NN::Module.new
    assert_equal true, mod.training
    assert_kind_of Hash, mod.state
  end
end
