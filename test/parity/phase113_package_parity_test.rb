# frozen_string_literal: true

require_relative "test_helper"

class Phase113PackageParityTest < Minitest::Test
def setup
  TestSupport.build_native_extension!
  $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
  require "mlx"
end

def teardown
  $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
end

  def test_phase_113_contract
    assert defined?(MLX::NN)
    assert defined?(MLX::Optimizers)
    assert defined?(MLX::DistributedUtils)
  end
end
