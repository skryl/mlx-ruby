# frozen_string_literal: true

require_relative "test_helper"

class Phase112PackageParityTest < Minitest::Test
def setup
  TestSupport.build_native_extension!
  $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
  require "mlx"
end

def teardown
  $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
end

  def test_phase_112_contract
    assert_respond_to MLX::DistributedUtils, :launch_mpi
    assert MLX::DistributedUtils.const_defined?(:CommandProcess)
    assert MLX::DistributedUtils.const_defined?(:RemoteProcess)
  end
end
