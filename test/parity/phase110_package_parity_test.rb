# frozen_string_literal: true

require_relative "test_helper"

class Phase110PackageParityTest < Minitest::Test
def setup
  TestSupport.build_native_extension!
  $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
  require "mlx"
end

def teardown
  $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
end

  def test_phase_110_contract
    assert_respond_to MLX::DistributedUtils, :positive_number
    assert MLX::DistributedUtils.const_defined?(:Host)
    assert MLX::DistributedUtils.const_defined?(:Hostfile)
  end
end
