# frozen_string_literal: true

require_relative "test_helper"

class Phase125PackageParityTest < Minitest::Test
def setup
  TestSupport.build_native_extension!
  $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
  require "mlx"
end

def teardown
  $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
end

  def test_phase_125_contract
    assert_kind_of MLX::Optimizers::Optimizer, MLX::Optimizers::Muon.new
  end
end
