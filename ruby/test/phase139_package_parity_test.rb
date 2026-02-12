# frozen_string_literal: true

require_relative "test_helper"

class Phase139PackageParityTest < Minitest::Test
def setup
  TestSupport.build_native_extension!
  $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
  require "mlx"
end

def teardown
  $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
end

  def test_phase_139_contract
    %i[Conv1d Conv2d Conv3d ConvTranspose1d ConvTranspose2d ConvTranspose3d MaxPool1d AvgPool1d MaxPool2d AvgPool2d MaxPool3d AvgPool3d].each { |n| assert MLX::NN.const_defined?(n) }
  end
end
