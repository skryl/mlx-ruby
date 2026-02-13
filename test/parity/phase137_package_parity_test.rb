# frozen_string_literal: true

require_relative "test_helper"

class Phase137PackageParityTest < Minitest::Test
def setup
  TestSupport.build_native_extension!
  $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
  require "mlx"
end

def teardown
  $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
end

  def test_phase_137_contract
    %i[sigmoid relu gelu tanh softmax].each { |n| assert_respond_to MLX::NN, n }
  end
end
